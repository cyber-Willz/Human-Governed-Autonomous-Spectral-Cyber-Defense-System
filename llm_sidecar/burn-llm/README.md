# burn-llm 0.3

Decoder-only LLM inference in Rust on [Burn](https://burn.dev) 0.21 (CPU/NdArray backend).
Families: **Llama, Mistral, Qwen2/2.5, Qwen3** (`config.json` `model_type`). Anything else is
rejected at load time with a message, not approximated.

```bash
cargo run --release --bin generate -- /path/to/Qwen3-0.6B --chat --no-think "Explain RoPE briefly."
cargo run --release --bin generate -- /path/to/model --check "some prompt"   # numerical health probe
```
Needs rustc **1.92+** (Burn 0.21's real MSRV; v0.2 claimed 1.85, which cargo refuses).

## What changed from v0.2 and why

Every item is either verified by a test in this crate or marked *unmeasured*.

**Correctness (silent-wrong-output bugs)**
| Area | v0.2 | v0.3 |
|---|---|---|
| Qwen3 `head_dim` | always `hidden/heads` (wrong: Qwen3-0.6B is 128, not 64) | explicit `head_dim` wins |
| Qwen3 QK-norm | not implemented | per-head RMSNorm before RoPE; presence checked against `model_type` both ways |
| RoPE scaling | none (Llama-3.1 and YaRN models decode wrongly past the original window) | `linear`, `llama3`, `yarn` per transformers v5.9.0; `dynamic`/`longrope`/per-layer-type rejected |
| transformers v5 configs | `rope_theta` read only top-level → silent fallback to 10000 | reads `rope_parameters` too |
| Stop tokens | one hard-coded `<|endoftext|>`/`</s>` (Qwen chat ends on `<|im_end|>`) | union of `config.json` + `generation_config.json`, then name fallback |
| Unsupported variants | Gemma/GELU, active sliding window, partial rotary loaded silently | hard error naming the field |
| Shapes | trusted | every tensor checked against config; unused tensors reported |
| LM head | trusted `tie_word_embeddings` | uses `lm_head.weight` if present, else ties |
| Streaming text | per-token decode (breaks UTF-8 / spaces) | incremental detokenizer seeded with prompt tail |

**Efficiency (structural; not benchmarked — no real checkpoint was available here)**
- GQA without expanding K/V: query heads are folded into the query axis, so the cache is never
  copied `group`× per layer per token.
- Logits only for the last position during generation (`forward_last`): skips `T × vocab` work in prefill.
- Causal mask built once per pass (none at all for single-token decode); RoPE table sized to the
  requested context instead of `max_position_embeddings`; chunked prefill bounds score memory.
- Loader builds layers at 1×1 and swaps in weights (no random init of every matrix); sharded
  checkpoints and f32/f16/bf16 supported.

**New:** ChatML/Llama-3 templating (`--chat`, Qwen3 `--no-think`), min-p and repetition penalty,
`generate::FinishReason`, `diagnostics::probe` (per-layer residual RMS / finiteness).

## Verification
`cargo test --release` — 40 tests. The important ones compare the real loader+model to an
**independent float64 numpy implementation** (`tools/make_fixtures.py`) at every position, through
full prefill, prefill+single-token decode, and chunked prefill, on three tiny checkpoints:
Qwen3 (explicit head_dim, QK-norm, GQA, YaRN, bf16) · Qwen2 (biases, tied head, GQA group 3,
Llama-3 RoPE) · Llama (plain MHA). Max |Δlogit| < 2e-3 is enforced. Fixtures are regenerable.

## Not verified / limitations
- Not run on a real pretrained checkpoint or against `transformers` itself (no weights/network
  to Hugging Face here). The numpy reference follows HF formulas but is the same author's reading
  of them; run `transformers` on a real model to close that gap.
- CPU only, no quantization, no paged/sliding KV cache, no batching beyond B=1 in the CLI,
  no tool-calling templates. `rust-version` 1.92 is required.
- The RoPE table is computed in f64 and rounded once; HF multiplies in f32, so very long contexts
  can differ from HF in the last bits.

## Relation to the nbsc / pde_shpinn archives
Those crates are spectral-graph / physics-informed-network research code (Dirichlet-energy
stabilisation of deep stacks, PDE training). They address *training-time* depth stability and PDE
solving, not inference of pretrained transformers, so no code was merged. The one idea carried
over is the monitoring habit: `diagnostics::probe` tracks the residual stream per layer so an
exploding/collapsing/non-finite stack is visible immediately.
