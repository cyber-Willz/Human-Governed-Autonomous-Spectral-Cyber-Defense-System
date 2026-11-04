use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::{backend::Backend, Tensor};

use crate::error::{HmmError, HmmResult};

/// Config for [`NeuralEmissionEngine`]. Kept separate from the module itself so it can be
/// serialized alongside a checkpoint and used to reconstruct an architecturally-identical
/// engine before loading weights into it.
#[derive(Debug, Clone)]
pub struct EmissionEngineConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_states: usize,
    pub dropout_prob: f64,
}

impl EmissionEngineConfig {
    pub fn new(input_dim: usize, num_states: usize) -> Self {
        Self { input_dim, hidden_dim: 64, num_states, dropout_prob: 0.1 }
    }
}

/// Two-hidden-layer MLP mapping a raw feature window to a probability distribution over
/// discrete HMM hidden states. Production differences from the prototype:
///   - an extra hidden layer + dropout for regularization,
///   - `log_softmax` output (numerically stable; avoids exponent overflow/underflow that
///     plain `softmax` followed by a later `.ln()` would hit),
///   - `forward` takes and validates batches, not a single hardcoded row.
#[derive(Module, Debug)]
pub struct NeuralEmissionEngine<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    dropout: Dropout,
    input_dim: usize,
    num_states: usize,
}

impl<B: Backend> NeuralEmissionEngine<B> {
    pub fn new(device: &B::Device, config: &EmissionEngineConfig) -> Self {
        let hidden2 = (config.hidden_dim / 2).max(config.num_states);
        Self {
            linear1: LinearConfig::new(config.input_dim, config.hidden_dim).init(device),
            linear2: LinearConfig::new(config.hidden_dim, hidden2).init(device),
            linear3: LinearConfig::new(hidden2, config.num_states).init(device),
            dropout: DropoutConfig::new(config.dropout_prob).init(),
            input_dim: config.input_dim,
            num_states: config.num_states,
        }
    }

    pub fn num_states(&self) -> usize {
        self.num_states
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Batched forward pass. `input` is `[batch, input_dim]`; returns `[batch, num_states]`
    /// log-probabilities (rows sum to 1 in probability space after `.exp()`).
    pub fn forward_log_probs(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = burn::tensor::activation::relu(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.linear3.forward(x);
        burn::tensor::activation::log_softmax(x, 1)
    }

    pub fn forward_probs(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_log_probs(input).exp()
    }

    /// Runs `forward_log_probs` and extracts row 0 into a plain `Vec<f32>`, validating shape
    /// and finiteness. This is the boundary between tensor-land and the plain-f32 HMM math,
    /// so it's where we guard against NaN/Inf leaking into the filter.
    pub fn emission_probs_single(&self, input: Tensor<B, 2>) -> HmmResult<Vec<f32>> {
        let [batch, dim] = input.dims();
        if batch == 0 {
            return Err(HmmError::EmptyBatch);
        }
        if dim != self.input_dim {
            return Err(HmmError::StateDimensionMismatch { expected: self.input_dim, actual: dim });
        }
        let log_probs = self.forward_log_probs(input);
        let probs = log_probs.exp();
        let data = probs.into_data().convert::<f32>();
        let flat = data.value;
        let row: Vec<f32> = flat.into_iter().take(self.num_states).collect();
        for (state, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(HmmError::NonFiniteValue { state, value: v });
            }
        }
        Ok(row)
    }

    pub fn save(&self, path: &str) -> HmmResult<()>
    where
        Self: Clone,
    {
        let recorder = CompactRecorder::new();
        recorder
            .record(self.clone().into_record(), path.into())
            .map_err(|e| HmmError::Checkpoint(e.to_string()))
    }

    pub fn load(self, path: &str, device: &B::Device) -> HmmResult<Self> {
        let recorder = CompactRecorder::new();
        let record = recorder
            .load(path.into(), device)
            .map_err(|e| HmmError::Checkpoint(e.to_string()))?;
        Ok(self.load_record(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn forward_produces_valid_distribution() {
        let device = Default::default();
        let config = EmissionEngineConfig::new(8, 3);
        let engine = NeuralEmissionEngine::<TestBackend>::new(&device, &config);
        let input = Tensor::<TestBackend, 2>::from_data([[0.1; 8]], &device);
        let probs = engine.emission_probs_single(input).unwrap();
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum was {sum}");
        assert!(probs.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn rejects_wrong_input_dim() {
        let device = Default::default();
        let config = EmissionEngineConfig::new(8, 3);
        let engine = NeuralEmissionEngine::<TestBackend>::new(&device, &config);
        let input = Tensor::<TestBackend, 2>::from_data([[0.1; 4]], &device);
        let err = engine.emission_probs_single(input).unwrap_err();
        assert!(matches!(err, HmmError::StateDimensionMismatch { .. }));
    }
}
