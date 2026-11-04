#!/usr/bin/env python3
"""Generate tiny HF-layout checkpoints + expected logits from an independent numpy reference.

The reference is float64, written separately from the Rust crate, and follows transformers'
formulas (modeling_llama / modeling_qwen2 / modeling_qwen3, modeling_rope_utils v5.9.0). The
Rust integration test (tests/reference_fixtures.rs) loads each fixture through the real
loader and compares logits position-by-position, through prefill, single-token decode and
chunked prefill.

    python3 tools/make_fixtures.py        # (re)writes tests/fixtures/*
"""
import json, math, os, struct
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")

# ---------------------------------------------------------------- safetensors writer
def bf16_round(a):
    """float32 -> bf16 (round-to-nearest-even) -> float32, returns (bytes, float32 values)."""
    u = np.ascontiguousarray(a.astype(np.float32)).view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    back = (rounded.astype(np.uint32) << 16).view(np.float32)
    return rounded.tobytes(), back

def write_safetensors(path, tensors, dtype):
    """tensors: name -> float32 array. Returns name -> the float32 values actually stored."""
    header, blobs, stored, off = {}, [], {}, 0
    for name in sorted(tensors):
        a = tensors[name].astype(np.float32)
        if dtype == "BF16":
            blob, vals = bf16_round(a)
        else:
            blob, vals = a.tobytes(), a
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + len(blob)]}
        blobs.append(blob); stored[name] = vals.astype(np.float64); off += len(blob)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for b in blobs: f.write(b)
    return stored

# ---------------------------------------------------------------- reference model (f64)
def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

def silu(x): return x / (1 + np.exp(-x))

def rope_inv_freq(cfg):
    d, base = cfg["head_dim"], cfg["rope_theta"]
    pos_freqs = base ** (np.arange(0, d, 2, dtype=np.float64) / d)
    inv = 1.0 / pos_freqs
    rs, af = cfg.get("rope_scaling"), 1.0
    if not rs: return inv, af
    t = rs["rope_type"]
    if t == "llama3":                       # HF _compute_llama3_parameters (torch.where form)
        f, lo, hi, old = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], rs["original_max_position_embeddings"]
        wl = 2 * math.pi / inv
        inv_l = np.where(wl > old / lo, inv / f, inv)
        smooth = (old / wl - lo) / (hi - lo)
        smoothed = (1 - smooth) * inv_l / f + smooth * inv_l
        medium = ~(wl < old / hi) * ~(wl > old / lo)
        return np.where(medium, smoothed, inv_l), 1.0
    if t == "yarn":                         # HF _compute_yarn_parameters
        f, old = rs["factor"], rs["original_max_position_embeddings"]
        bf, bs = rs.get("beta_fast", 32), rs.get("beta_slow", 1)
        corr = lambda n: (d * math.log(old / (n * 2 * math.pi))) / (2 * math.log(base))
        low, high = max(math.floor(corr(bf)), 0), min(math.ceil(corr(bs)), d - 1)
        if low == high: high += 0.001
        ramp = np.clip((np.arange(d // 2, dtype=np.float64) - low) / (high - low), 0, 1)
        extrap = 1 - ramp
        inv = (1.0 / (f * pos_freqs)) * (1 - extrap) + (1.0 / pos_freqs) * extrap
        return inv, 0.1 * math.log(f) + 1.0
    raise ValueError(t)

def rot_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)

def reference_logits(cfg, W, ids):
    T, H, KV, D, eps = len(ids), cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    inv, af = rope_inv_freq(cfg)
    ang = np.outer(np.arange(T), inv); emb = np.concatenate([ang, ang], -1)
    cos, sin = np.cos(emb) * af, np.sin(emb) * af
    x = W["model.embed_tokens.weight"][ids]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms(x, W[p + "input_layernorm.weight"], eps)
        def proj(n, heads):
            y = h @ W[p + f"self_attn.{n}_proj.weight"].T
            b = W.get(p + f"self_attn.{n}_proj.bias")
            return (y + b if b is not None else y).reshape(T, heads, D)
        q, k, v = proj("q", H), proj("k", KV), proj("v", KV)
        if cfg["model_type"] == "qwen3":
            q = rms(q, W[p + "self_attn.q_norm.weight"], eps)
            k = rms(k, W[p + "self_attn.k_norm.weight"], eps)
        q = q * cos[:, None, :] + rot_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + rot_half(k) * sin[:, None, :]
        out = np.zeros((T, H, D))
        for hh in range(H):
            kv = hh // (H // KV)                         # HF repeat_kv mapping
            s = q[:, hh] @ k[:, kv].T / math.sqrt(D) + mask
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            out[:, hh] = s @ v[:, kv]
        x = x + out.reshape(T, H * D) @ W[p + "self_attn.o_proj.weight"].T
        h = rms(x, W[p + "post_attention_layernorm.weight"], eps)
        x = x + (silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
    x = rms(x, W["model.norm.weight"], eps)
    return x @ W.get("lm_head.weight", W["model.embed_tokens.weight"]).T

# ---------------------------------------------------------------- fixtures
def make(name, cfg, dtype, seed, tied, bias):
    rng = np.random.default_rng(seed)
    V, Hd, I, L = cfg["vocab_size"], cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    n = lambda *s, sc=0.15: rng.normal(0, sc, s).astype(np.float32)
    g = lambda *s: (1 + rng.normal(0, 0.2, s)).astype(np.float32)   # non-trivial norm gains
    t = {"model.embed_tokens.weight": n(V, Hd, sc=0.5), "model.norm.weight": g(Hd)}
    if not tied: t["lm_head.weight"] = n(V, Hd, sc=0.3)
    for i in range(L):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = g(Hd); t[p + "post_attention_layernorm.weight"] = g(Hd)
        t[p + "self_attn.q_proj.weight"] = n(H * D, Hd); t[p + "self_attn.k_proj.weight"] = n(KV * D, Hd)
        t[p + "self_attn.v_proj.weight"] = n(KV * D, Hd); t[p + "self_attn.o_proj.weight"] = n(Hd, H * D)
        if bias:
            for nme, sz in (("q", H * D), ("k", KV * D), ("v", KV * D)):
                t[p + f"self_attn.{nme}_proj.bias"] = n(sz, sc=0.3)
        if cfg["model_type"] == "qwen3":
            t[p + "self_attn.q_norm.weight"] = g(D); t[p + "self_attn.k_norm.weight"] = g(D)
        t[p + "mlp.gate_proj.weight"] = n(I, Hd); t[p + "mlp.up_proj.weight"] = n(I, Hd); t[p + "mlp.down_proj.weight"] = n(Hd, I)
    d = os.path.join(ROOT, name); os.makedirs(d, exist_ok=True)
    stored = write_safetensors(os.path.join(d, "model.safetensors"), t, dtype)
    if name.startswith("qwen3"):          # CLI smoke-test extras (do not affect the logits)
        cfg = dict(cfg, eos_token_id=2)
        vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
        vocab.update({f"w{i}": i + 3 for i in range(V - 3)})
        specials = [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                     "normalized": False, "special": True} for c, i in list(vocab.items())[:3]]
        json.dump({"version": "1.0", "truncation": None, "padding": None, "added_tokens": specials,
                   "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": None,
                   "decoder": None, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>"}},
                  open(os.path.join(d, "tokenizer.json"), "w"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    ids = [int(v) for v in rng.integers(0, V, 9)]
    logits = reference_logits(cfg, stored, ids)
    json.dump({"ids": ids, "logits": logits.tolist()}, open(os.path.join(d, "expected.json"), "w"))
    print(f"{name}: dtype={dtype} logits {logits.shape} |max|={np.abs(logits).max():.3f}")

if __name__ == "__main__":
    # Qwen3: explicit head_dim (16) != hidden/heads (8), GQA 4:2, QK-norm, YaRN, bf16, untied head.
    make("qwen3_yarn_bf16", dict(
        model_type="qwen3", vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
        max_position_embeddings=64, hidden_act="silu", tie_word_embeddings=False, rope_theta=10000.0,
        rope_scaling=dict(rope_type="yarn", factor=4.0, original_max_position_embeddings=16)),
        "BF16", 1, tied=False, bias=False)
    # Qwen2: q/k/v biases, tied head, GQA 6:2 (group 3), Llama-3.1 three-band RoPE, f32.
    make("qwen2_llama3rope_f32", dict(
        model_type="qwen2", vocab_size=40, hidden_size=48, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=6, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
        max_position_embeddings=64, hidden_act="silu", tie_word_embeddings=True, rope_theta=500000.0,
        rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                          original_max_position_embeddings=16)),
        "F32", 2, tied=True, bias=True)
    # Llama: plain MHA (kv == heads), untied head, no scaling, f16-free f32.
    make("llama_mha_f32", dict(
        model_type="llama", vocab_size=36, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=3, num_key_value_heads=3, rms_norm_eps=1e-5, head_dim=8,
        max_position_embeddings=64, hidden_act="silu", tie_word_embeddings=False, rope_theta=10000.0),
        "F32", 3, tied=False, bias=False)
