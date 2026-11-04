use burn::module::Module;
use burn::nn::{Embedding, Linear};
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::attention::{build_causal_mask, KvCache};
use crate::block::DecoderBlock;
use crate::config::{ModelConfig, RopeConfig};
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryCache;

#[derive(Module, Debug)]
pub struct LlmModel<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<DecoderBlock<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    /// Plain inference-time config (no tensors); skipped by the `Module` derive.
    #[module(skip)]
    pub config: ModelConfigOwned,
}

/// Plain-data subset of [`ModelConfig`] needed at inference time.
#[derive(Debug, Clone)]
pub struct ModelConfigOwned {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub rope: RopeConfig,
}

impl From<&ModelConfig> for ModelConfigOwned {
    fn from(c: &ModelConfig) -> Self {
        Self {
            num_heads: c.num_attention_heads,
            num_kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            max_position_embeddings: c.max_position_embeddings,
            vocab_size: c.vocab_size,
            rope: c.rope.clone(),
        }
    }
}

/// Per-session state: one KV cache per layer, the RoPE table, and the next position.
pub struct GenerationState<B: Backend> {
    pub caches: Vec<KvCache<B>>,
    pub rope: RotaryCache<B>,
    pub position: usize,
}

impl<B: Backend> GenerationState<B> {
    /// Total positions (prompt + generated) this state can hold.
    pub fn capacity(&self) -> usize {
        self.rope.capacity()
    }
    pub fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.position)
    }
}

impl<B: Backend> LlmModel<B> {
    pub fn new(
        embed_tokens: Embedding<B>,
        layers: Vec<DecoderBlock<B>>,
        norm: RmsNorm<B>,
        lm_head: Linear<B>,
        config: ModelConfigOwned,
    ) -> Self {
        Self { embed_tokens, layers, norm, lm_head, config }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `max_seq`: positions to reserve (prompt + new tokens). The RoPE table is built for
    /// exactly this many positions, so ask for what you need, not `max_position_embeddings`.
    pub fn init_generation(&self, device: &B::Device, max_seq: usize) -> GenerationState<B> {
        GenerationState {
            caches: (0..self.layers.len()).map(|_| KvCache::new()).collect(),
            rope: RotaryCache::new(self.config.head_dim, max_seq.max(1), &self.config.rope, device),
            position: 0,
        }
    }

    /// Runs all decoder blocks; returns the pre-final-norm hidden state `[B, T, hidden]`.
    /// If `probe` is given, the residual stream after every block is appended to it.
    fn hidden(
        &self,
        token_ids: Tensor<B, 2, Int>,
        state: &mut GenerationState<B>,
        mut probe: Option<&mut Vec<Tensor<B, 3>>>,
    ) -> Tensor<B, 3> {
        let [_, t] = token_ids.dims();
        assert!(
            state.position + t <= state.capacity(),
            "sequence would reach position {} but the state was initialised for {} positions",
            state.position + t,
            state.capacity()
        );
        let device = token_ids.device();
        let mut h = self.embed_tokens.forward(token_ids);

        // One mask per forward pass, shared by every layer (v0.2 rebuilt it per layer).
        let group = self.config.num_heads / self.config.num_kv_heads;
        let mask = build_causal_mask::<B>(group, t, state.position + t, state.position, &device);

        for (layer, cache) in self.layers.iter().zip(state.caches.iter_mut()) {
            h = layer.forward(h, &state.rope, state.position, mask.as_ref(), cache);
            if let Some(p) = probe.as_mut() {
                p.push(h.clone());
            }
        }
        state.position += t;
        h
    }

    /// Logits for **every** position, `[B, T, vocab]`. Use [`Self::forward_last`] for generation.
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>, state: &mut GenerationState<B>) -> Tensor<B, 3> {
        let h = self.hidden(token_ids, state, None);
        self.lm_head.forward(self.norm.forward(h))
    }

    /// Logits for the **last** position only, `[B, vocab]`.
    ///
    /// Slices before the final norm + `lm_head`, so a prompt of `T` tokens costs one
    /// `hidden × vocab` matmul instead of `T` of them. For a 152k-vocab model this avoids
    /// materialising `T × 152k × 4 B` of logits (≈311 MB for a 512-token chunk) that
    /// v0.2 computed and immediately discarded.
    pub fn forward_last(
        &self,
        token_ids: Tensor<B, 2, Int>,
        state: &mut GenerationState<B>,
    ) -> Tensor<B, 2> {
        let h = self.hidden(token_ids, state, None);
        self.last_logits(h)
    }

    /// Like [`Self::forward_last`] but also returns the residual stream after each block
    /// (for [`crate::diagnostics`]).
    pub fn forward_last_probe(
        &self,
        token_ids: Tensor<B, 2, Int>,
        state: &mut GenerationState<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 3>>) {
        let mut taps = Vec::with_capacity(self.layers.len());
        let h = self.hidden(token_ids, state, Some(&mut taps));
        (self.last_logits(h), taps)
    }

    fn last_logits(&self, h: Tensor<B, 3>) -> Tensor<B, 2> {
        let [b, t, hid] = h.dims();
        let last = h.slice([0..b, t - 1..t, 0..hid]);
        let logits = self.lm_head.forward(self.norm.forward(last)); // [B, 1, vocab]
        let v = logits.dims()[2];
        logits.reshape([b, v])
    }
}
