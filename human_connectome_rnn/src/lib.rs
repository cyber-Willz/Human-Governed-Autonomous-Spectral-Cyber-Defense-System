//! # human_connectome_rnn
//!
//! Recurrent neural networks constrained by a human structural connectome:
//! wiring from tractography, conduction delays from fibre length, Dale's-law
//! excitatory/inhibitory populations per region, trained with exact BPTT, with
//! Arnoldi-based stability control (`krylov_ds`), a connectome-Laplacian energy
//! regulariser and harmonic analysis (`spectral_hypergraph`) and non-backtracking
//! topology diagnostics (`nbsc`).
//!
//! ```no_run
//! use hcrnn::{Connectome, ConnectomeRnn, RnnConfig, MemoryGuidedTask, TrainConfig};
//! let conn = Connectome::synthetic_human(40, 0.25, 1).unwrap().log_compressed();
//! let (inp, out) = hcrnn::network::default_roles(&conn).unwrap();
//! let cfg = RnnConfig { input_regions: inp, output_regions: out, ..Default::default() };
//! let mut net = ConnectomeRnn::new(&conn, cfg).unwrap();
//! let report = hcrnn::train::train(&mut net, &MemoryGuidedTask::default(), &TrainConfig::default(), |_| {});
//! println!("{:?}", report.final_eval);
//! ```

pub mod connectome;
pub mod lstm;
pub mod model;
pub mod network;
pub mod optim;
pub mod rng;
pub mod spectral;
pub mod task;
pub mod topology;
pub mod train;

pub use connectome::{Connectome, Region};
pub use lstm::{ConnectomeLstm, LstmConfig};
pub use model::SequenceModel;
pub use network::{Calibration, ConnectomeRnn, MaskMode, RnnConfig};
pub use task::{MemoryGuidedTask, Trial};
pub use train::{EvalMetrics, TrainConfig};
