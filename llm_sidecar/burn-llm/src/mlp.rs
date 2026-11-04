use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::activation::silu;
use burn::tensor::{backend::Backend, Tensor};

/// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(gate_proj: Linear<B>, up_proj: Linear<B>, down_proj: Linear<B>) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}
