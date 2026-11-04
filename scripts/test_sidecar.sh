#!/usr/bin/env sh
# Builds the burn-llm sidecar (separate workspace) and runs the llm_bridge
# test that drives the real model through it.
set -e
cd "$(dirname "$0")/.."
cargo build --release --manifest-path llm_sidecar/Cargo.toml -p llm_sidecar
LLM_SIDECAR_BIN="$PWD/llm_sidecar/target/release/llm_sidecar" cargo test -p llm_bridge -- --nocapture
