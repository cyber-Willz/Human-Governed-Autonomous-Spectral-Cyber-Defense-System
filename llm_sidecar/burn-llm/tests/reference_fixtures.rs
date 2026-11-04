//! Cross-validation against an independent float64 numpy reference (tools/make_fixtures.py).
use std::path::PathBuf;

use burn::tensor::{Int, Tensor};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_llm::diagnostics;
use burn_llm::generate::{generate, SamplingConfig};
use burn_llm::model::LlmModel;
use burn_llm::weights::load_model;
use rand::rngs::StdRng;
use rand::SeedableRng;

type B = NdArray<f32>;

fn dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

struct Fixture {
    model: LlmModel<B>,
    ids: Vec<u32>,
    expected: Vec<Vec<f64>>, // [T][V]
    device: NdArrayDevice,
}

fn load(name: &str) -> Fixture {
    let device = NdArrayDevice::default();
    let model = load_model::<B>(dir(name), None, &device).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir(name).join("expected.json")).unwrap()).unwrap();
    let ids = v["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let expected = v["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())
        .collect();
    Fixture { model, ids, expected, device }
}

fn ids_t(ids: &[u32], d: &NdArrayDevice) -> Tensor<B, 2, Int> {
    let v: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    Tensor::<B, 1, Int>::from_ints(v.as_slice(), d).reshape([1, ids.len()])
}

fn row(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len());
    let worst = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
    assert!(worst < 2e-3, "{what}: max |rust - numpy| = {worst:.3e}");
}

fn check_all_paths(name: &str) {
    let f = load(name);
    let t = f.ids.len();

    // 1. full prefill: logits at every position
    let mut st = f.model.init_generation(&f.device, t);
    let all = f.model.forward(ids_t(&f.ids, &f.device), &mut st);
    let v = all.dims()[2];
    let flat = all.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    for p in 0..t {
        assert_close(&flat[p * v..(p + 1) * v], &f.expected[p], &format!("{name} prefill pos {p}"));
    }

    // 2. prefill 4 tokens then single-token decode (KV cache + offset RoPE + mask-free path)
    let mut st = f.model.init_generation(&f.device, t);
    let l = row(f.model.forward_last(ids_t(&f.ids[..4], &f.device), &mut st));
    assert_close(&l, &f.expected[3], &format!("{name} prefill(4) last"));
    for p in 4..t {
        let l = row(f.model.forward_last(ids_t(&f.ids[p..p + 1], &f.device), &mut st));
        assert_close(&l, &f.expected[p], &format!("{name} decode pos {p}"));
    }

    // 3. chunked prefill with a multi-token chunk on a non-empty cache (offset causal mask)
    let mut st = f.model.init_generation(&f.device, t);
    let _ = f.model.forward_last(ids_t(&f.ids[..3], &f.device), &mut st);
    let l = row(f.model.forward_last(ids_t(&f.ids[3..7], &f.device), &mut st));
    assert_close(&l, &f.expected[6], &format!("{name} chunk [3..7)"));
    let l = row(f.model.forward_last(ids_t(&f.ids[7..], &f.device), &mut st));
    assert_close(&l, &f.expected[t - 1], &format!("{name} chunk [7..)"));

    // 4. residual-stream probe is healthy
    assert!(diagnostics::probe(&f.model, &f.ids, &f.device).healthy());
}

#[test]
fn qwen3_yarn_bf16_matches_numpy() {
    check_all_paths("qwen3_yarn_bf16");
}

#[test]
fn qwen2_bias_tied_llama3_rope_matches_numpy() {
    check_all_paths("qwen2_llama3rope_f32");
}

#[test]
fn llama_mha_matches_numpy() {
    check_all_paths("llama_mha_f32");
}

#[test]
fn greedy_generation_is_deterministic_and_chunk_invariant() {
    let f = load("qwen2_llama3rope_f32");
    let mut rng = StdRng::seed_from_u64(0);
    let mut run = |chunk: usize| {
        let cfg = SamplingConfig { max_new_tokens: 12, prefill_chunk: chunk, ..SamplingConfig::greedy() };
        generate(&f.model, &f.ids, &[], &cfg, &f.device, &mut rng, |_| {}).unwrap().sequence
    };
    let a = run(512);
    assert_eq!(a.len(), f.ids.len() + 12);
    assert_eq!(a, run(512));
    assert_eq!(a, run(2), "chunked prefill must not change greedy output");
    assert_eq!(a, run(1));
}

#[test]
fn generation_stops_on_a_stop_id_and_reports_it() {
    let f = load("llama_mha_f32");
    let mut rng = StdRng::seed_from_u64(0);
    let free = generate(&f.model, &f.ids, &[], &SamplingConfig { max_new_tokens: 6, ..SamplingConfig::greedy() }, &f.device, &mut rng, |_| {}).unwrap();
    let first = free.sequence[f.ids.len()];
    let mut streamed = 0;
    let stopped = generate(&f.model, &f.ids, &[first], &SamplingConfig { max_new_tokens: 6, ..SamplingConfig::greedy() }, &f.device, &mut rng, |_| streamed += 1).unwrap();
    assert_eq!(stopped.new_tokens, 1);
    assert_eq!(streamed, 0, "the stop token itself is not streamed");
    assert_eq!(stopped.finish, burn_llm::generate::FinishReason::Stop);
}

#[test]
fn qwen3_without_qk_norm_tensors_is_refused() {
    // Copy the qwen3 fixture but relabel it llama: q_norm tensors present, model_type disagrees.
    let tmp = std::env::temp_dir().join(format!("burn_llm_qknorm_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(dir("qwen3_yarn_bf16").join("model.safetensors"), tmp.join("model.safetensors")).unwrap();
    let cfg = std::fs::read_to_string(dir("qwen3_yarn_bf16").join("config.json"))
        .unwrap()
        .replace("\"qwen3\"", "\"llama\"");
    std::fs::write(tmp.join("config.json"), cfg).unwrap();
    let err = load_model::<B>(&tmp, None, &NdArrayDevice::default()).err().expect("must fail").to_string();
    assert!(err.contains("q_norm"), "{err}");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn shape_mismatch_between_config_and_checkpoint_is_reported() {
    let tmp = std::env::temp_dir().join(format!("burn_llm_shape_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(dir("llama_mha_f32").join("model.safetensors"), tmp.join("model.safetensors")).unwrap();
    let cfg = std::fs::read_to_string(dir("llama_mha_f32").join("config.json"))
        .unwrap()
        .replace("\"intermediate_size\": 48", "\"intermediate_size\": 50");
    std::fs::write(tmp.join("config.json"), cfg).unwrap();
    let err = load_model::<B>(&tmp, None, &NdArrayDevice::default()).err().expect("must fail").to_string();
    assert!(err.contains("config/checkpoint mismatch"), "{err}");
    let _ = std::fs::remove_dir_all(&tmp);
}
