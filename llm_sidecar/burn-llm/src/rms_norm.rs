use burn::module::{Module, Param};
use burn::tensor::{backend::Backend, Tensor};

/// RMSNorm over the **last** dimension: `x / sqrt(mean(x²) + eps) * weight`.
///
/// Generic over rank so the same module serves the `[B, T, hidden]` block norms and
/// Qwen3's per-head QK-norm on `[B, T, heads, head_dim]` (weight shape `[head_dim]`).
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn from_weight(weight: Tensor<B, 1>, eps: f64) -> Self {
        Self { weight: Param::from_tensor(weight), eps }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let n = x.dims()[D - 1];
        let variance = x.clone().powf_scalar(2.0).mean_dim(D - 1); // keeps the reduced dim: [.., 1]
        let x = x / (variance + self.eps).sqrt();
        let mut shape = [1usize; D];
        shape[D - 1] = n;
        x * self.weight.val().reshape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type B = NdArray<f32>;

    fn vals<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
    }

    #[test]
    fn unit_weight_normalizes_rms_to_one() {
        let device = NdArrayDevice::default();
        let norm = RmsNorm::from_weight(Tensor::<B, 1>::ones([4], &device), 1e-6);
        let out = vals(norm.forward(Tensor::<B, 3>::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &device)));
        let rms = 7.5f32.sqrt(); // sqrt(mean([1,4,9,16]))
        for (a, x) in out.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((a - x / rms).abs() < 1e-5, "{a} vs {}", x / rms);
        }
    }

    #[test]
    fn rank4_normalizes_each_head_independently_with_shared_weight() {
        let device = NdArrayDevice::default();
        // [B=1, T=1, H=2, D=2]; weight [2, 3] broadcast over both heads.
        let w = Tensor::<B, 1>::from_floats([2.0, 3.0], &device);
        let norm = RmsNorm::from_weight(w, 0.0);
        let x = Tensor::<B, 4>::from_floats([[[[3.0, 4.0], [0.0, 5.0]]]], &device);
        let out = vals(norm.forward(x));
        // head 0: rms = sqrt((9+16)/2)=3.5355 ; head 1: rms = sqrt(25/2)=3.5355
        let r = (12.5f32).sqrt();
        let want = [3.0 / r * 2.0, 4.0 / r * 3.0, 0.0, 5.0 / r * 3.0];
        for (a, b) in out.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
