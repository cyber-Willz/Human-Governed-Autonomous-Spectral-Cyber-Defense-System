use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};

use crate::attention::{Attention, KvCache};
use crate::mlp::Mlp;
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryCache;

/// One decoder layer: pre-norm attention + residual, then pre-norm SwiGLU MLP + residual.
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    input_layernorm: RmsNorm<B>,
    self_attn: Attention<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(
        input_layernorm: RmsNorm<B>,
        self_attn: Attention<B>,
        post_attention_layernorm: RmsNorm<B>,
        mlp: Mlp<B>,
    ) -> Self {
        Self { input_layernorm, self_attn, post_attention_layernorm, mlp }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryCache<B>,
        rope_offset: usize,
        mask: Option<&Tensor<B, 4>>,
        cache: &mut KvCache<B>,
    ) -> Tensor<B, 3> {
        let h = self.input_layernorm.forward(x.clone());
        let x = x + self.self_attn.forward(h, rope, rope_offset, mask, cache);
        let h = self.post_attention_layernorm.forward(x.clone());
        x + self.mlp.forward(h)
    }
}
