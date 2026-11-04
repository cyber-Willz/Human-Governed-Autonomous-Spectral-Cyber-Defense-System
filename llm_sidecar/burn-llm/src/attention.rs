use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::{backend::Backend, Tensor};

use crate::rms_norm::RmsNorm;
use crate::rope::{apply_rope, RotaryCache};

/// One layer's KV cache: `[B, kv_heads, T_so_far, head_dim]`.
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    fn append(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (k_full, v_full) = match (self.k.take(), self.v.take()) {
            (Some(kp), Some(vp)) => (Tensor::cat(vec![kp, k], 2), Tensor::cat(vec![vp, v], 2)),
            _ => (k, v),
        };
        self.k = Some(k_full.clone());
        self.v = Some(v_full.clone());
        (k_full, v_full)
    }

    pub fn len(&self) -> usize {
        self.k.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<B: Backend> Default for KvCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// GQA causal self-attention (optionally with Qwen3-style per-head QK-norm).
///
/// **Grouped layout.** Instead of expanding K/V from `kv_heads` to `num_heads` (a
/// `group`-times larger copy of the whole cache, per layer, per generated token), the
/// `group = num_heads / kv_heads` query heads that share a KV head are folded into the
/// query-length axis: `q: [B, H, T, D] → [B, KV, group·T, D]`. Scores are then
/// `[B, KV, group·T, T_kv]` from a plain batched matmul against the *unexpanded* K. Query head
/// `h` still uses KV head `h / group` (HF's `repeat_kv` mapping), because `h = kv·group + g`.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: Linear<B>,
        k_proj: Linear<B>,
        v_proj: Linear<B>,
        o_proj: Linear<B>,
        q_norm: Option<RmsNorm<B>>,
        k_norm: Option<RmsNorm<B>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        assert!(num_heads % num_kv_heads == 0);
        assert_eq!(q_norm.is_some(), k_norm.is_some(), "q_norm and k_norm come as a pair");
        Self { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, num_heads, num_kv_heads, head_dim }
    }

    /// `x`: `[B, T, hidden]`. `mask`: additive `[1, 1, group·T, T_kv]` from
    /// [`build_causal_mask`], or `None` for single-token decode (every cached key is visible).
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryCache<B>,
        rope_offset: usize,
        mask: Option<&Tensor<B, 4>>,
        cache: &mut KvCache<B>,
    ) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let (h, kv, d) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let q = self.q_proj.forward(x.clone()).reshape([b, t, h, d]);
        let k = self.k_proj.forward(x.clone()).reshape([b, t, kv, d]);
        let v = self.v_proj.forward(x).reshape([b, t, kv, d]);

        // Qwen3: RMSNorm over head_dim, *before* RoPE.
        let q = match &self.q_norm {
            Some(n) => n.forward(q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(n) => n.forward(k),
            None => k,
        };

        let q = q.swap_dims(1, 2); // [B, H, T, D]
        let k = k.swap_dims(1, 2); // [B, KV, T, D]
        let v = v.swap_dims(1, 2);

        let (cos, sin) = rope.slice(rope_offset, t);
        let q = apply_rope(q, &cos, &sin);
        let k = apply_rope(k, &cos, &sin);

        let (k, v) = cache.append(k, v); // [B, KV, T_kv, D]

        let group = h / kv;
        let q = q.reshape([b, kv, group * t, d]);

        let scale = (d as f64).powf(-0.5);
        let mut scores = q.matmul(k.swap_dims(2, 3)) * scale; // [B, KV, group·T, T_kv]
        if let Some(m) = mask {
            scores = scores + m.clone();
        }
        let out = softmax(scores, 3).matmul(v); // [B, KV, group·T, D]

        let out = out
            .reshape([b, h, t, d])
            .swap_dims(1, 2)
            .reshape([b, t, h * d]);
        self.o_proj.forward(out)
    }
}

/// Additive causal mask for the grouped layout, shape `[1, 1, group·q_len, kv_len]`.
///
/// Row `r` of the folded query axis is query position `r % q_len` (rows are ordered
/// `[group][position]`), whose absolute position is `offset + r % q_len`; it may attend to key
/// `j` iff `j <= offset + r % q_len`.
///
/// Returns `None` for `q_len == 1`: a single new token sees the entire cache, so no mask is
/// needed (v0.2 rebuilt and added an all-zero mask in every layer on every decode step).
pub fn build_causal_mask<B: Backend>(
    group: usize,
    q_len: usize,
    kv_len: usize,
    offset: usize,
    device: &B::Device,
) -> Option<Tensor<B, 4>> {
    if q_len == 1 {
        return None;
    }
    let rows = group * q_len;
    let mut data = vec![0f32; rows * kv_len];
    for r in 0..rows {
        let abs_pos = offset + r % q_len;
        for j in (abs_pos + 1)..kv_len {
            data[r * kv_len + j] = f32::NEG_INFINITY;
        }
    }
    Some(Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 1, rows, kv_len]))
}

/// Builds a Burn `Linear` from an already-transposed `[in, out]` weight.
///
/// HF safetensors store `[out, in]`; `weights.rs` transposes before calling this.
///
/// The config-built layer is created at `1 x 1` and then has its parameters *replaced*:
/// `LinearConfig::init` runs a random initializer over the whole weight, which for a real
/// checkpoint is a full pass of RNG draws (and a transient extra copy of every matrix) that
/// gets overwritten immediately. `Linear::forward` only reads the parameter tensors.
pub fn linear_from_weight<B: Backend>(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Linear<B> {
    let mut linear = LinearConfig::new(1, 1).with_bias(false).init::<B>(&weight.device());
    linear.weight = burn::module::Param::from_tensor(weight);
    linear.bias = bias.map(burn::module::Param::from_tensor);
    linear
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type B = NdArray<f32>;

    fn mask_vec(m: Tensor<B, 4>) -> Vec<f32> {
        m.into_data().convert::<f32>().to_vec::<f32>().unwrap()
    }

    #[test]
    fn decode_step_needs_no_mask() {
        let device = NdArrayDevice::default();
        assert!(build_causal_mask::<B>(7, 1, 100, 99, &device).is_none());
    }

    #[test]
    fn prefill_mask_is_causal_and_repeats_per_group() {
        let device = NdArrayDevice::default();
        let (group, t) = (2, 3);
        let m = mask_vec(build_causal_mask::<B>(group, t, t, 0, &device).unwrap());
        for r in 0..group * t {
            let pos = r % t;
            for j in 0..t {
                let masked = m[r * t + j].is_infinite();
                assert_eq!(masked, j > pos, "row {r} (pos {pos}) col {j}");
            }
        }
    }

    #[test]
    fn chunked_prefill_mask_respects_offset() {
        // 2 new tokens appended after 3 cached ones (kv_len = 5), group 1.
        let device = NdArrayDevice::default();
        let m = mask_vec(build_causal_mask::<B>(1, 2, 5, 3, &device).unwrap());
        // query 0 has absolute position 3 -> sees keys 0..=3, not 4
        assert!(m[0..4].iter().all(|x| *x == 0.0) && m[4].is_infinite());
        // query 1 has absolute position 4 -> sees all 5
        assert!(m[5..10].iter().all(|x| *x == 0.0));
    }
}
