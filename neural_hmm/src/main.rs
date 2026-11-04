use burn::backend::NdArray;
use burn::tensor::Tensor;

use neural_hmm::{Belief, EmissionEngineConfig, NeuralEmissionEngine, NeuralHmm, TransitionMatrix};

// NdArray is a CPU backend: no GPU driver dependency, deterministic, fast to build under
// Rust 1.75 via apt. Swap in burn::backend::Wgpu (or a candle/tch backend) at this single
// type alias if/when GPU acceleration is worth the extra build weight.
type EngineBackend = NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let device = Default::default();

    // 3 states: [0: Normal, 1: Suspicious, 2: Attack]
    let transition_matrix = TransitionMatrix::with_labels(
        vec![
            vec![0.85, 0.12, 0.03],
            vec![0.10, 0.70, 0.20],
            vec![0.05, 0.05, 0.90],
        ],
        vec!["normal".into(), "suspicious".into(), "attack".into()],
    )?;

    let emission_config = EmissionEngineConfig::new(16, transition_matrix.num_states());
    let emission_engine = NeuralEmissionEngine::<EngineBackend>::new(&device, &emission_config);

    let hmm = NeuralHmm::new(emission_engine, transition_matrix)?;

    let mut belief = Belief::certain(hmm.num_states(), 0)?; // start certain we're "normal"

    let mock_features = Tensor::<EngineBackend, 2>::from_data(
        [[
            0.23, 1.4, 0.0, 0.9, 0.12, 0.44, 0.89, 0.11, 0.0, 0.1, 0.2, 0.5, 0.7, 1.2, 0.1, 0.0,
        ]],
        &device,
    );

    belief = hmm.filter_step(&belief, mock_features)?;

    let (state, prob) = belief.argmax();
    println!("Updated belief: {:?}", belief.as_slice());
    println!("Most likely state: {state} (p={prob:.4})");

    Ok(())
}
