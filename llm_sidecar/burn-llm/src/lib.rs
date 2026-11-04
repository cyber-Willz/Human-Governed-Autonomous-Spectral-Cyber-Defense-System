//! `burn-llm` — decoder-only LLM inference on Burn 0.21.
//!
//! Supported families (`config.json` `model_type`): `llama`, `mistral`, `qwen2`, `qwen3`.
//! Anything else is rejected at load time instead of being approximated.

pub mod attention;
pub mod block;
pub mod chat;
pub mod config;
pub mod diagnostics;
pub mod generate;
pub mod mlp;
pub mod model;
pub mod rms_norm;
pub mod rope;
pub mod tokenizer;
pub mod weights;
