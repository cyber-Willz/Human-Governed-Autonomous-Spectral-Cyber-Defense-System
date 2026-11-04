use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use burn::module::Param;
use burn::nn::EmbeddingConfig;
use burn::tensor::{backend::Backend, Tensor};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::attention::{linear_from_weight, Attention};
use crate::block::DecoderBlock;
use crate::config::ModelConfig;
use crate::mlp::Mlp;
use crate::model::{LlmModel, ModelConfigOwned};
use crate::rms_norm::RmsNorm;

/// Raw tensor bytes → f32 regardless of on-disk dtype.
fn to_f32_vec(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        Dtype::F16 => Ok(data
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()),
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()),
        other => Err(anyhow!(
            "unsupported safetensors dtype {other:?} (quantized checkpoints are not supported; \
             use f32/f16/bf16)"
        )),
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// A checkpoint possibly split over several memory-mapped shards.
struct Weights<'a> {
    shards: Vec<SafeTensors<'a>>,
    /// tensor name → shard index (`None`: single file, everything is in shard 0).
    tensor_shard: Option<HashMap<String, usize>>,
    /// Every tensor name seen in the checkpoint, for "unused tensor" reporting.
    used: std::cell::RefCell<HashSet<String>>,
}

impl<'a> Weights<'a> {
    fn shard_for(&self, name: &str) -> Result<&SafeTensors<'a>> {
        match &self.tensor_shard {
            Some(map) => {
                let idx = map
                    .get(name)
                    .with_context(|| format!("tensor `{name}` not listed in checkpoint index"))?;
                Ok(&self.shards[*idx])
            }
            None => self.shards.first().ok_or_else(|| anyhow!("no checkpoint shards loaded")),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.shard_for(name).ok().is_some_and(|s| s.tensor(name).is_ok())
    }

    fn all_names(&self) -> Vec<String> {
        match &self.tensor_shard {
            Some(m) => m.keys().cloned().collect(),
            None => self.shards.first().map(|s| s.names().into_iter().map(str::to_owned).collect()).unwrap_or_default(),
        }
    }

    fn get_raw(&self, name: &str, expect: &[usize]) -> Result<Vec<f32>> {
        let view = self
            .shard_for(name)?
            .tensor(name)
            .with_context(|| format!("missing tensor `{name}` in checkpoint"))?;
        if view.shape() != expect {
            bail!(
                "tensor `{name}` has shape {:?} but config.json implies {expect:?} \
                 (config/checkpoint mismatch — wrong config.json, or an unsupported variant)",
                view.shape()
            );
        }
        self.used.borrow_mut().insert(name.to_owned());
        to_f32_vec(view.dtype(), view.data())
    }

    /// `[out, in]` on disk → `[in, out]` for Burn's `Linear`.
    fn linear_weight<B: Backend>(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let data = self.get_raw(name, &[out_dim, in_dim])?;
        Ok(Tensor::<B, 1>::from_floats(data.as_slice(), device)
            .reshape([out_dim, in_dim])
            .transpose())
    }

    fn vector<B: Backend>(&self, name: &str, len: usize, device: &B::Device) -> Result<Tensor<B, 1>> {
        Ok(Tensor::<B, 1>::from_floats(self.get_raw(name, &[len])?.as_slice(), device))
    }

    fn matrix<B: Backend>(&self, name: &str, r: usize, c: usize, device: &B::Device) -> Result<Tensor<B, 2>> {
        Ok(Tensor::<B, 1>::from_floats(self.get_raw(name, &[r, c])?.as_slice(), device).reshape([r, c]))
    }
}

enum CheckpointLayout {
    SingleFile(PathBuf),
    Sharded { index_path: PathBuf, weight_map: HashMap<String, String> },
}

fn resolve_layout(dir: &Path, weights_filename: Option<&str>) -> Result<CheckpointLayout> {
    if let Some(name) = weights_filename {
        return Ok(CheckpointLayout::SingleFile(dir.join(name)));
    }
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let s = fs::read_to_string(&index_path).with_context(|| format!("reading {index_path:?}"))?;
        let index: SafetensorsIndex =
            serde_json::from_str(&s).with_context(|| format!("parsing {index_path:?}"))?;
        if index.weight_map.is_empty() {
            bail!("{index_path:?} has an empty weight_map");
        }
        return Ok(CheckpointLayout::Sharded { index_path, weight_map: index.weight_map });
    }
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(CheckpointLayout::SingleFile(single));
    }
    bail!(
        "neither model.safetensors.index.json nor model.safetensors found in {dir:?} — pass the \
         weights filename explicitly"
    )
}

fn mmap_shards(
    dir: &Path,
    layout: &CheckpointLayout,
) -> Result<(Vec<PathBuf>, Vec<Mmap>, HashMap<String, usize>)> {
    let shard_paths: Vec<PathBuf> = match layout {
        CheckpointLayout::SingleFile(p) => vec![p.clone()],
        CheckpointLayout::Sharded { weight_map, .. } => {
            let mut names: Vec<String> = weight_map.values().cloned().collect();
            names.sort();
            names.dedup();
            names.into_iter().map(|f| dir.join(f)).collect()
        }
    };
    let mut mmaps = Vec::with_capacity(shard_paths.len());
    let mut by_name = HashMap::new();
    for (i, p) in shard_paths.iter().enumerate() {
        let f = File::open(p).with_context(|| format!("opening shard {p:?}"))?;
        // SAFETY: checkpoint files are treated as immutable for the duration of loading.
        mmaps.push(unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {p:?}"))?);
        if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
            by_name.insert(n.to_owned(), i);
        }
    }
    Ok((shard_paths, mmaps, by_name))
}

/// Loads a Llama / Mistral / Qwen2 / Qwen3 model from a HF-layout directory.
///
/// Every decision that depends on what the checkpoint contains is made from **tensor
/// presence**, never from a config flag that isn't standardised across families:
///
/// * q/k/v biases (Qwen2) — present ⇔ `q_proj.bias` exists; all three must then exist.
/// * QK-norm (Qwen3) — present ⇔ `q_norm.weight` exists. If `model_type` says `qwen3` but the
///   tensor is missing (or the reverse), loading fails: running without a norm the weights
///   were trained with is silently wrong output.
/// * LM head — uses `lm_head.weight` if present, otherwise ties to the embedding, regardless
///   of `tie_word_embeddings` (some exports set the flag inconsistently with the file).
///
/// Every tensor shape is checked against the config, and tensors the loader did not consume
/// (other than known-ignorable ones such as `rotary_emb.inv_freq`) are reported on stderr.
pub fn load_model<B: Backend>(
    dir: impl AsRef<Path>,
    weights_filename: Option<&str>,
    device: &B::Device,
) -> Result<LlmModel<B>> {
    let dir = dir.as_ref();
    let config_str = fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("reading config.json in {dir:?}"))?;
    let config = ModelConfig::from_json_str(&config_str)?;

    let layout = resolve_layout(dir, weights_filename)?;
    let (paths, mmaps, by_name) = mmap_shards(dir, &layout)?;
    let mut shards = Vec::with_capacity(mmaps.len());
    for (p, m) in paths.iter().zip(mmaps.iter()) {
        shards.push(SafeTensors::deserialize(m).with_context(|| format!("parsing shard {p:?}"))?);
    }
    let tensor_shard = match &layout {
        CheckpointLayout::SingleFile(_) => None,
        CheckpointLayout::Sharded { weight_map, index_path } => {
            let mut map = HashMap::with_capacity(weight_map.len());
            for (t, f) in weight_map {
                let idx = by_name.get(f).copied().with_context(|| {
                    format!("{index_path:?} maps `{t}` to shard `{f}`, which wasn't loaded")
                })?;
                map.insert(t.clone(), idx);
            }
            Some(map)
        }
    };
    let w = Weights { shards, tensor_shard, used: Default::default() };

    let (hid, inter) = (config.hidden_size, config.intermediate_size);
    let (nh, nkv, hd) = (config.num_attention_heads, config.num_key_value_heads, config.head_dim);
    let (q_out, kv_out) = (nh * hd, nkv * hd);

    // Embedding: build at 1x1 and replace the parameter (no random-init pass over vocab×hidden).
    let embed_weight = w.matrix::<B>("model.embed_tokens.weight", config.vocab_size, hid, device)?;
    let mut embed_tokens = EmbeddingConfig::new(1, 1).init::<B>(device);
    embed_tokens.weight = Param::from_tensor(embed_weight.clone());

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let eps = config.rms_norm_eps;

        let input_ln = RmsNorm::from_weight(w.vector::<B>(&format!("{p}.input_layernorm.weight"), hid, device)?, eps);
        let post_ln =
            RmsNorm::from_weight(w.vector::<B>(&format!("{p}.post_attention_layernorm.weight"), hid, device)?, eps);

        // --- q/k/v bias: decided by the checkpoint, all-or-nothing ---
        let bias_names = ["q", "k", "v"].map(|n| format!("{p}.self_attn.{n}_proj.bias"));
        let present: Vec<bool> = bias_names.iter().map(|n| w.has(n)).collect();
        if present.iter().any(|x| *x) && !present.iter().all(|x| *x) {
            bail!("layer {i}: only some of the q/k/v projection biases are present in the checkpoint");
        }
        let has_bias = present[0];
        let bias = |name: &str, len: usize| -> Result<Option<Tensor<B, 1>>> {
            has_bias.then(|| w.vector::<B>(name, len, device)).transpose()
        };
        let q_proj = linear_from_weight(
            w.linear_weight::<B>(&format!("{p}.self_attn.q_proj.weight"), q_out, hid, device)?,
            bias(&bias_names[0], q_out)?,
        );
        let k_proj = linear_from_weight(
            w.linear_weight::<B>(&format!("{p}.self_attn.k_proj.weight"), kv_out, hid, device)?,
            bias(&bias_names[1], kv_out)?,
        );
        let v_proj = linear_from_weight(
            w.linear_weight::<B>(&format!("{p}.self_attn.v_proj.weight"), kv_out, hid, device)?,
            bias(&bias_names[2], kv_out)?,
        );
        let o_proj = linear_from_weight(
            w.linear_weight::<B>(&format!("{p}.self_attn.o_proj.weight"), hid, q_out, device)?,
            None,
        );

        // --- QK-norm: checkpoint presence must agree with the model family ---
        let qn = format!("{p}.self_attn.q_norm.weight");
        let kn = format!("{p}.self_attn.k_norm.weight");
        let (has_qn, has_kn) = (w.has(&qn), w.has(&kn));
        if has_qn != has_kn {
            bail!("layer {i}: q_norm/k_norm must both be present or both absent");
        }
        if config.qk_norm && !has_qn {
            bail!(
                "model_type is qwen3 but layer {i} has no `self_attn.q_norm.weight`; refusing to \
                 run Qwen3 without the QK-norm it was trained with"
            );
        }
        if !config.qk_norm && has_qn {
            bail!(
                "layer {i} has q_norm/k_norm tensors but config.json's model_type ({:?}) does not \
                 declare QK-norm; refusing to ignore them",
                config.model_type
            );
        }
        let (q_norm, k_norm) = if has_qn {
            (
                Some(RmsNorm::from_weight(w.vector::<B>(&qn, hd, device)?, eps)),
                Some(RmsNorm::from_weight(w.vector::<B>(&kn, hd, device)?, eps)),
            )
        } else {
            (None, None)
        };

        let self_attn = Attention::new(q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, nh, nkv, hd);

        let mlp = Mlp::new(
            linear_from_weight(w.linear_weight::<B>(&format!("{p}.mlp.gate_proj.weight"), inter, hid, device)?, None),
            linear_from_weight(w.linear_weight::<B>(&format!("{p}.mlp.up_proj.weight"), inter, hid, device)?, None),
            linear_from_weight(w.linear_weight::<B>(&format!("{p}.mlp.down_proj.weight"), hid, inter, device)?, None),
        );
        layers.push(DecoderBlock::new(input_ln, self_attn, post_ln, mlp));
    }

    let norm = RmsNorm::from_weight(w.vector::<B>("model.norm.weight", hid, device)?, config.rms_norm_eps);

    let lm_head_weight = if w.has("lm_head.weight") {
        w.linear_weight::<B>("lm_head.weight", config.vocab_size, hid, device)?
    } else {
        if !config.tie_word_embeddings {
            eprintln!(
                "note: no lm_head.weight in the checkpoint although tie_word_embeddings=false; \
                 tying the output head to the embedding"
            );
        }
        embed_weight.transpose()
    };
    let lm_head = linear_from_weight(lm_head_weight, None);

    // Report anything in the checkpoint we did not consume.
    let used = w.used.borrow();
    let mut unused: Vec<String> = w
        .all_names()
        .into_iter()
        .filter(|n| !used.contains(n) && !n.ends_with("rotary_emb.inv_freq"))
        .collect();
    unused.sort();
    if !unused.is_empty() {
        let shown: Vec<&str> = unused.iter().take(6).map(String::as_str).collect();
        eprintln!(
            "warning: {} checkpoint tensor(s) were not used by this architecture, e.g. {:?}",
            unused.len(),
            shown
        );
    }

    Ok(LlmModel::new(embed_tokens, layers, norm, lm_head, ModelConfigOwned::from(&config)))
}
