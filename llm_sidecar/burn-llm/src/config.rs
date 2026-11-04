//! HuggingFace `config.json` → [`ModelConfig`].
//!
//! Design rule (carried over from v0.2's `attention_bias` bug): **never let a missing or
//! renamed field silently fall back to a default that changes the math.** Anything this
//! crate cannot reproduce faithfully (an unsupported `model_type`, a non-SiLU activation,
//! active sliding-window attention, per-layer-type RoPE, partial rotary, …) is a hard
//! error naming the offending field, not a quiet approximation.

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

/// `model_type` values whose decoder layer this crate implements.
///
/// * `llama`, `mistral` – pre-norm GQA decoder, SwiGLU, RoPE.
/// * `qwen2` – the same plus q/k/v projection biases.
/// * `qwen3` – the same plus per-head QK-RMSNorm and an *explicit* `head_dim`.
pub const SUPPORTED_MODEL_TYPES: &[&str] = &["llama", "mistral", "qwen2", "qwen3"];

/// RoPE frequency-scaling variant, mirroring `rope_type` in transformers'
/// `modeling_rope_utils.py` (v5.9.0). `dynamic`, `longrope` and `proportional` are
/// rejected at parse time rather than approximated.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// Plain RoPE: `inv_freq_i = theta^(-2i/d)`.
    None,
    /// `inv_freq /= factor` (position interpolation).
    Linear { factor: f64 },
    /// Llama-3.1 three-band scheme.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: usize,
    },
    /// YaRN (NTK-by-parts + attention temperature).
    Yarn {
        factor: f64,
        original_max_position_embeddings: usize,
        beta_fast: f64,
        beta_slow: f64,
        /// `None` → `0.1*ln(factor)+1` (HF default).
        attention_factor: Option<f64>,
        truncate: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    pub theta: f64,
    pub scaling: RopeScaling,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self { theta: 10000.0, scaling: RopeScaling::None }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Resolved: equals `num_attention_heads` when the field is absent (plain MHA).
    pub num_key_value_heads: usize,
    /// Resolved: the explicit `head_dim` field if present (Qwen3, Mistral-Nemo, …),
    /// otherwise `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub rope: RopeConfig,
    /// `true` for `model_type == "qwen3"`. The loader treats tensor presence
    /// (`self_attn.q_norm.weight`) as authoritative and errors if the two disagree
    /// in the dangerous direction (config says QK-norm, checkpoint has none).
    pub qk_norm: bool,
    /// `eos_token_id` from config.json (int or list). May be empty; see
    /// [`crate::tokenizer::resolve_stop_ids`] for the full stop-token resolution.
    pub eos_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
}

// ----- small JSON helpers -------------------------------------------------

fn get_usize(v: &Value, k: &str) -> Option<usize> {
    v.get(k)?.as_u64().map(|x| x as usize)
}
fn get_f64(v: &Value, k: &str) -> Option<f64> {
    v.get(k)?.as_f64()
}
fn req_usize(v: &Value, k: &str) -> Result<usize> {
    get_usize(v, k).ok_or_else(|| anyhow!("config.json: missing or non-integer field `{k}`"))
}
fn req_f64(v: &Value, k: &str) -> Result<f64> {
    get_f64(v, k).ok_or_else(|| anyhow!("config.json: missing or non-numeric field `{k}`"))
}

/// Accepts `5`, `[5, 7]`, `null`, or a missing key.
pub fn id_list(v: Option<&Value>) -> Vec<u32> {
    match v {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect(),
        _ => Vec::new(),
    }
}

fn parse_rope(v: &Value, max_pos: usize) -> Result<RopeConfig> {
    // transformers >= 5 writes `rope_parameters` (with `rope_theta` *inside* it);
    // older files use top-level `rope_theta` + optional `rope_scaling`. Read both.
    let theta_top = get_f64(v, "rope_theta");
    let rp = v
        .get("rope_parameters")
        .filter(|x| x.is_object())
        .or_else(|| v.get("rope_scaling").filter(|x| x.is_object()));

    let Some(r) = rp else {
        return Ok(RopeConfig { theta: theta_top.unwrap_or(10000.0), scaling: RopeScaling::None });
    };

    let has_type = r.get("rope_type").is_some() || r.get("type").is_some();
    if !has_type && r.as_object().is_some_and(|o| o.values().any(|x| x.is_object())) {
        bail!(
            "config.json: per-layer-type `rope_parameters` (Gemma-3 style nested dict) is not \
             supported; refusing to silently pick one layer type's RoPE"
        );
    }
    if let Some(p) = get_f64(r, "partial_rotary_factor") {
        if (p - 1.0).abs() > 1e-12 {
            bail!("config.json: partial_rotary_factor={p} is not supported (full rotary only)");
        }
    }
    if r.get("mscale").is_some() || r.get("mscale_all_dim").is_some() {
        bail!("config.json: rope `mscale`/`mscale_all_dim` (DeepSeek-style YaRN) is not supported");
    }

    let theta = get_f64(r, "rope_theta").or(theta_top).unwrap_or(10000.0);
    let ty = r
        .get("rope_type")
        .or_else(|| r.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    let orig = get_usize(r, "original_max_position_embeddings").unwrap_or(max_pos);

    let scaling = match ty {
        "default" => RopeScaling::None,
        "linear" => RopeScaling::Linear { factor: req_f64(r, "factor")? },
        "llama3" => {
            let low = req_f64(r, "low_freq_factor")?;
            let high = req_f64(r, "high_freq_factor")?;
            if (high - low).abs() < 1e-12 {
                bail!("config.json: llama3 rope needs high_freq_factor != low_freq_factor");
            }
            RopeScaling::Llama3 {
                factor: req_f64(r, "factor")?,
                low_freq_factor: low,
                high_freq_factor: high,
                original_max_position_embeddings: orig,
            }
        }
        "yarn" => RopeScaling::Yarn {
            factor: req_f64(r, "factor")?,
            original_max_position_embeddings: orig,
            // HF: `rope_parameters.get("beta_fast") or 32` (0/None → default)
            beta_fast: get_f64(r, "beta_fast").filter(|x| *x != 0.0).unwrap_or(32.0),
            beta_slow: get_f64(r, "beta_slow").filter(|x| *x != 0.0).unwrap_or(1.0),
            attention_factor: get_f64(r, "attention_factor"),
            truncate: r.get("truncate").and_then(Value::as_bool).unwrap_or(true),
        },
        other => bail!(
            "config.json: rope_type `{other}` is not supported (supported: default, linear, \
             llama3, yarn)"
        ),
    };
    Ok(RopeConfig { theta, scaling })
}

impl ModelConfig {
    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(s)?;
        Self::from_value(&v)
    }

    pub fn from_value(root: &Value) -> Result<Self> {
        // Multimodal wrappers nest the LM under `text_config`.
        let v = if root.get("hidden_size").is_none() {
            root.get("text_config").unwrap_or(root)
        } else {
            root
        };

        let model_type = v.get("model_type").and_then(Value::as_str).map(str::to_owned);
        if let Some(mt) = &model_type {
            if !SUPPORTED_MODEL_TYPES.contains(&mt.as_str()) {
                bail!(
                    "config.json: model_type `{mt}` is not supported (supported: {}). \
                     Loading it anyway would produce fluent-looking but wrong output.",
                    SUPPORTED_MODEL_TYPES.join(", ")
                );
            }
        }

        // The MLP here is SwiGLU. gelu-family activations (Gemma) must not load silently.
        for key in ["hidden_act", "hidden_activation"] {
            if let Some(act) = v.get(key).and_then(Value::as_str) {
                if act != "silu" {
                    bail!("config.json: {key}=`{act}` is not supported (only `silu`/SwiGLU)");
                }
            }
        }

        let vocab_size = req_usize(v, "vocab_size")?;
        let hidden_size = req_usize(v, "hidden_size")?;
        let intermediate_size = req_usize(v, "intermediate_size")?;
        let num_hidden_layers = req_usize(v, "num_hidden_layers")?;
        let num_attention_heads = req_usize(v, "num_attention_heads")?;
        let max_position_embeddings = get_usize(v, "max_position_embeddings").unwrap_or(4096);

        let num_key_value_heads = get_usize(v, "num_key_value_heads")
            .filter(|&k| k != 0)
            .unwrap_or(num_attention_heads);
        if num_key_value_heads == 0 || num_attention_heads % num_key_value_heads != 0 {
            bail!(
                "config.json: num_attention_heads ({num_attention_heads}) must be a multiple of \
                 num_key_value_heads ({num_key_value_heads})"
            );
        }

        // Sliding-window attention: only an error if it would actually be active.
        // Qwen2.5 ships `sliding_window: 131072` with `use_sliding_window: false` (inactive);
        // Mistral-7B-v0.1 ships `sliding_window: 4096` and no flag (active).
        if let Some(w) = get_usize(v, "sliding_window") {
            let enabled = v.get("use_sliding_window").and_then(Value::as_bool).unwrap_or(true);
            if enabled && w < max_position_embeddings {
                bail!(
                    "config.json: active sliding-window attention (sliding_window={w}) is not \
                     supported"
                );
            }
        }

        // Explicit head_dim wins (Qwen3: hidden 1024 / 16 heads would give 64, but the
        // real head_dim is 128). Deriving it from hidden/heads was a v0.2 bug for these models.
        let head_dim = match get_usize(v, "head_dim") {
            Some(d) if d > 0 => d,
            _ => {
                if hidden_size % num_attention_heads != 0 {
                    bail!(
                        "config.json: hidden_size ({hidden_size}) is not divisible by \
                         num_attention_heads ({num_attention_heads}) and no head_dim is given"
                    );
                }
                hidden_size / num_attention_heads
            }
        };
        if head_dim % 2 != 0 {
            bail!("config.json: head_dim ({head_dim}) must be even for RoPE");
        }

        let rope = parse_rope(v, max_position_embeddings)?;
        let qk_norm = model_type.as_deref() == Some("qwen3");

        Ok(Self {
            model_type,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: get_f64(v, "rms_norm_eps").unwrap_or(1e-6),
            max_position_embeddings,
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            rope,
            qk_norm,
            eos_token_ids: id_list(v.get("eos_token_id")),
            bos_token_id: id_list(v.get("bos_token_id")).first().copied(),
        })
    }

    /// Kept for source compatibility with v0.2 (`cfg.head_dim()` / `cfg.kv_heads()`).
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads
    }
    /// Query heads per KV head (GQA group size).
    pub fn gqa_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shaped like Qwen2.5-0.5B: GQA 14:2, tied embeddings, theta 1e6, and the
    // `sliding_window` + `use_sliding_window:false` pair that must NOT be rejected.
    const QWEN25: &str = r#"{
        "model_type": "qwen2", "vocab_size": 151936, "hidden_size": 896,
        "intermediate_size": 4864, "num_hidden_layers": 24, "num_attention_heads": 14,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
        "max_position_embeddings": 32768, "tie_word_embeddings": true,
        "hidden_act": "silu", "sliding_window": 32768, "use_sliding_window": false,
        "eos_token_id": 151643
    }"#;

    // Shaped like Qwen3-0.6B: hidden/heads = 64 but the real head_dim is 128.
    const QWEN3: &str = r#"{
        "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 1024,
        "intermediate_size": 3072, "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000, "max_position_embeddings": 40960,
        "tie_word_embeddings": true, "hidden_act": "silu",
        "sliding_window": null, "use_sliding_window": false
    }"#;

    #[test]
    fn qwen2_style_config() {
        let c = ModelConfig::from_json_str(QWEN25).unwrap();
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.kv_heads(), 2);
        assert_eq!(c.gqa_group(), 7);
        assert!(c.tie_word_embeddings);
        assert!(!c.qk_norm);
        assert_eq!(c.rope.theta, 1_000_000.0);
        assert_eq!(c.eos_token_ids, vec![151643]);
    }

    #[test]
    fn qwen3_explicit_head_dim_is_not_hidden_over_heads() {
        let c = ModelConfig::from_json_str(QWEN3).unwrap();
        assert_eq!(c.hidden_size / c.num_attention_heads, 64, "the wrong (v0.2) derivation");
        assert_eq!(c.head_dim, 128, "explicit head_dim must win");
        assert!(c.qk_norm);
    }

    #[test]
    fn defaults_to_mha_when_kv_heads_absent() {
        let c = ModelConfig::from_json_str(
            r#"{"vocab_size":32000,"hidden_size":4096,"intermediate_size":11008,
                "num_hidden_layers":32,"num_attention_heads":32}"#,
        )
        .unwrap();
        assert_eq!(c.kv_heads(), 32);
        assert_eq!(c.rope, RopeConfig::default());
    }

    #[test]
    fn llama31_rope_scaling_parsed() {
        let c = ModelConfig::from_json_str(
            r#"{"model_type":"llama","vocab_size":128256,"hidden_size":4096,
                "intermediate_size":14336,"num_hidden_layers":32,"num_attention_heads":32,
                "num_key_value_heads":8,"max_position_embeddings":131072,
                "rope_theta":500000.0,"eos_token_id":[128001,128008,128009],
                "rope_scaling":{"factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,
                                "original_max_position_embeddings":8192,"rope_type":"llama3"}}"#,
        )
        .unwrap();
        assert_eq!(c.rope.theta, 500000.0);
        assert_eq!(
            c.rope.scaling,
            RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192
            }
        );
        assert_eq!(c.eos_token_ids, vec![128001, 128008, 128009]);
    }

    #[test]
    fn transformers_v5_nested_rope_parameters_carry_theta() {
        // v5 configs put rope_theta *inside* rope_parameters. Reading only the top-level key
        // would silently fall back to 10000 – the same failure class as the v0.2 bias bug.
        let c = ModelConfig::from_json_str(
            r#"{"model_type":"qwen3","vocab_size":100,"hidden_size":64,"intermediate_size":128,
                "num_hidden_layers":2,"num_attention_heads":4,"head_dim":32,
                "rope_parameters":{"rope_type":"default","rope_theta":1000000.0}}"#,
        )
        .unwrap();
        assert_eq!(c.rope.theta, 1_000_000.0);
    }

    #[test]
    fn yarn_parsed_with_hf_defaults() {
        let c = ModelConfig::from_json_str(
            r#"{"model_type":"qwen2","vocab_size":100,"hidden_size":64,"intermediate_size":128,
                "num_hidden_layers":2,"num_attention_heads":4,"max_position_embeddings":32768,
                "rope_theta":1000000.0,
                "rope_scaling":{"type":"yarn","factor":4.0,
                                "original_max_position_embeddings":32768}}"#,
        )
        .unwrap();
        match c.rope.scaling {
            RopeScaling::Yarn { factor, beta_fast, beta_slow, attention_factor, truncate, .. } => {
                assert_eq!(factor, 4.0);
                assert_eq!((beta_fast, beta_slow), (32.0, 1.0));
                assert_eq!(attention_factor, None);
                assert!(truncate);
            }
            other => panic!("expected yarn, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_things_fail_loudly() {
        let base = |extra: &str| {
            format!(
                r#"{{"vocab_size":100,"hidden_size":64,"intermediate_size":128,
                    "num_hidden_layers":2,"num_attention_heads":4,
                    "max_position_embeddings":8192 {extra}}}"#
            )
        };
        let err = |extra: &str| ModelConfig::from_json_str(&base(extra)).unwrap_err().to_string();

        assert!(err(r#","model_type":"gemma3""#).contains("gemma3"));
        assert!(err(r#","hidden_act":"gelu_pytorch_tanh""#).contains("gelu"));
        assert!(err(r#","sliding_window":4096"#).contains("sliding-window"));
        assert!(err(r#","rope_scaling":{"rope_type":"longrope","factor":2.0}"#).contains("longrope"));
        assert!(err(
            r#","rope_parameters":{"full_attention":{"rope_type":"default"},"sliding_attention":{"rope_type":"default"}}"#
        )
        .contains("per-layer"));
        assert!(err(r#","rope_parameters":{"rope_type":"default","partial_rotary_factor":0.5}"#)
            .contains("partial_rotary"));
        assert!(err(r#","num_key_value_heads":3"#).contains("multiple"));
    }

    #[test]
    fn id_list_accepts_int_list_null() {
        assert_eq!(id_list(Some(&serde_json::json!(7))), vec![7]);
        assert_eq!(id_list(Some(&serde_json::json!([1, 2]))), vec![1, 2]);
        assert!(id_list(Some(&serde_json::json!(null))).is_empty());
        assert!(id_list(None).is_empty());
    }
}
