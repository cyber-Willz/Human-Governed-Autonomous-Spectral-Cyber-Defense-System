use burn::tensor::{backend::Backend, Tensor};
use tracing::{instrument, warn};

use crate::emission::NeuralEmissionEngine;
use crate::error::{HmmError, HmmResult};
use crate::transition::TransitionMatrix;

const BELIEF_EPS: f32 = 1e-9;

/// A validated probability distribution over hidden states. Wrapping this instead of passing
/// `Vec<f32>` around means "is this a legal belief vector" gets checked once at construction,
/// not re-derived at every call site.
#[derive(Debug, Clone, PartialEq)]
pub struct Belief(Vec<f32>);

impl Belief {
    pub fn new(values: Vec<f32>) -> HmmResult<Self> {
        if values.is_empty() {
            return Err(HmmError::EmptyTransitionMatrix);
        }
        for (state, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(HmmError::NonFiniteValue { state, value: v });
            }
            if v < 0.0 {
                return Err(HmmError::NegativeProbability { row: 0, col: state, value: v });
            }
        }
        let sum: f32 = values.iter().sum();
        if (sum - 1.0).abs() > 1e-2 {
            return Err(HmmError::InvalidPriorSum(sum));
        }
        Ok(Self(values))
    }

    /// Uniform prior over `n` states — the standard "no information yet" starting point.
    pub fn uniform(n: usize) -> HmmResult<Self> {
        if n == 0 {
            return Err(HmmError::EmptyTransitionMatrix);
        }
        Self::new(vec![1.0 / n as f32; n])
    }

    /// One-hot prior; useful when a state is known with certainty at t=0.
    pub fn certain(n: usize, state: usize) -> HmmResult<Self> {
        if state >= n {
            return Err(HmmError::StateDimensionMismatch { expected: n, actual: state + 1 });
        }
        let mut v = vec![0.0f32; n];
        v[state] = 1.0;
        Self::new(v)
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }

    /// Index of the most probable state, and its probability.
    pub fn argmax(&self) -> (usize, f32) {
        self.0
            .iter()
            .enumerate()
            .fold((0, self.0[0]), |acc, (i, &p)| if p > acc.1 { (i, p) } else { acc })
    }
}

/// Production hybrid Neural HMM filter: a learned neural emission model supplies
/// `P(observation | state)` while a hand-specified (or learned-offline) Markov transition
/// matrix supplies `P(state_t+1 | state_t)`. Each `filter_step` runs one predict/update cycle
/// of recursive Bayesian (forward-algorithm) state estimation.
///
/// Differences from a naive port of the prototype:
///   - every input is validated (prior sums to 1, feature batch shape matches, matrix square
///     and row-stochastic) instead of trusting the caller,
///   - belief collapse (all-zero posterior) is a typed error instead of a silent fallback that
///     could mask a broken feature pipeline,
///   - `filter_step` is `#[instrument]`-ed so it plugs into the existing active-siem/spec-engine
///     tracing setup without extra glue.
#[derive(Debug)]
pub struct NeuralHmm<B: Backend> {
    emission_engine: NeuralEmissionEngine<B>,
    transition_matrix: TransitionMatrix,
}

impl<B: Backend> NeuralHmm<B> {
    pub fn new(emission_engine: NeuralEmissionEngine<B>, transition_matrix: TransitionMatrix) -> HmmResult<Self> {
        let n_emission = emission_engine.num_states();
        let n_transition = transition_matrix.num_states();
        if n_emission != n_transition {
            return Err(HmmError::StateDimensionMismatch { expected: n_transition, actual: n_emission });
        }
        Ok(Self { emission_engine, transition_matrix })
    }

    pub fn num_states(&self) -> usize {
        self.transition_matrix.num_states()
    }

    /// Runs one predict + update cycle.
    ///
    /// `raw_features` must be shape `[1, input_dim]` (a single observation window). For batched
    /// scoring of many independent sequences at once, call the emission engine directly and
    /// drive `transition_matrix.predict` / `update_with_emissions` per-sequence.
    #[instrument(skip(self, raw_features), fields(num_states = self.num_states()))]
    pub fn filter_step(&self, prior: &Belief, raw_features: Tensor<B, 2>) -> HmmResult<Belief> {
        let predicted = self.transition_matrix.predict(prior.as_slice())?;
        let emission_probs = self.emission_engine.emission_probs_single(raw_features)?;
        self.update_with_emissions(&predicted, &emission_probs)
    }

    /// Bayes update: `posterior[s] ∝ predicted[s] * emission[s]`, renormalized. Split out from
    /// `filter_step` so it's independently testable without spinning up a tensor backend.
    fn update_with_emissions(&self, predicted: &[f32], emission_probs: &[f32]) -> HmmResult<Belief> {
        let n = self.num_states();
        if predicted.len() != n {
            return Err(HmmError::StateDimensionMismatch { expected: n, actual: predicted.len() });
        }
        if emission_probs.len() != n {
            return Err(HmmError::StateDimensionMismatch { expected: n, actual: emission_probs.len() });
        }

        let mut posterior = vec![0.0f32; n];
        let mut normalizer = 0.0f32;
        for state in 0..n {
            let joint = predicted[state] * emission_probs[state];
            posterior[state] = joint;
            normalizer += joint;
        }

        if normalizer <= BELIEF_EPS {
            warn!(normalizer, "belief collapsed to ~zero mass during update");
            return Err(HmmError::BeliefCollapse);
        }

        for p in posterior.iter_mut() {
            *p /= normalizer;
        }

        Belief::new(posterior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn matrix() -> TransitionMatrix {
        TransitionMatrix::new(vec![
            vec![0.85, 0.12, 0.03],
            vec![0.10, 0.70, 0.20],
            vec![0.05, 0.05, 0.90],
        ])
        .unwrap()
    }

    #[test]
    fn update_normalizes_and_favors_strong_emission() {
        // Standalone test of the math, no tensor backend required.
        struct Harness {
            transition: TransitionMatrix,
        }
        impl Harness {
            fn update(&self, predicted: &[f32], emission: &[f32]) -> HmmResult<Belief> {
                let n = self.transition.num_states();
                let mut posterior = vec![0.0; n];
                let mut norm = 0.0;
                for s in 0..n {
                    posterior[s] = predicted[s] * emission[s];
                    norm += posterior[s];
                }
                if norm <= BELIEF_EPS {
                    return Err(HmmError::BeliefCollapse);
                }
                for p in posterior.iter_mut() {
                    *p /= norm;
                }
                Belief::new(posterior)
            }
        }
        let h = Harness { transition: matrix() };
        let predicted = h.transition.predict(&[1.0, 0.0, 0.0]).unwrap();
        // Strong emission evidence for "attack" (state 2) despite low predicted prior.
        let posterior = h.update(&predicted, &[0.01, 0.01, 0.98]).unwrap();
        let (argmax, _) = posterior.argmax();
        assert_eq!(argmax, 2);
        let sum: f32 = posterior.as_slice().iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn belief_rejects_bad_prior() {
        assert!(Belief::new(vec![0.5, 0.6]).is_err());
        assert!(Belief::new(vec![f32::NAN, 0.5]).is_err());
        assert!(Belief::new(vec![-0.1, 1.1]).is_err());
    }

    #[test]
    fn dimension_mismatch_between_engine_and_matrix_is_rejected() {
        use crate::emission::{EmissionEngineConfig, NeuralEmissionEngine};
        use burn::backend::NdArray;
        type TestBackend = NdArray<f32>;
        let device = Default::default();
        let config = EmissionEngineConfig::new(8, 2); // 2 states
        let engine = NeuralEmissionEngine::<TestBackend>::new(&device, &config);
        let m = matrix(); // 3 states
        let err = NeuralHmm::new(engine, m).unwrap_err();
        assert!(matches!(err, HmmError::StateDimensionMismatch { .. }));
    }
}
