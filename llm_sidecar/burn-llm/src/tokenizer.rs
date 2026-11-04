use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use tokenizers::Tokenizer as HfTokenizer;

use crate::config::{id_list, ModelConfig};

/// Special-token names that end a turn/sequence in the model families this crate supports.
/// Only used as a *fallback* when neither config.json nor generation_config.json name an EOS.
const STOP_TOKEN_NAMES: [&str; 5] =
    ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "</s>"];

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = HfTokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow!("failed to load tokenizer.json: {e}"))?;
        Ok(Self { inner })
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let inner = json
            .parse::<HfTokenizer>()
            .map_err(|e| anyhow!("failed to parse tokenizer json: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow!("tokenizer decode failed: {e}"))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// First fallback stop token found by name (v0.2 API; prefer [`resolve_stop_ids`]).
    pub fn eos_token_id(&self) -> Option<u32> {
        STOP_TOKEN_NAMES.iter().find_map(|n| self.token_to_id(n))
    }
}

pub fn load_tokenizer(dir: impl AsRef<Path>) -> Result<Tokenizer> {
    let path = dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(&path).with_context(|| format!("loading tokenizer from {path:?}"))
}

/// Every token id that should end generation.
///
/// Union of `eos_token_id` in `config.json` and in `generation_config.json` (each may be an
/// int or a list — chat-tuned Qwen2.5 lists both `<|im_end|>` and `<|endoftext|>`), falling
/// back to well-known stop-token names only if neither file names one. v0.2 hard-coded a
/// single `<|endoftext|>`/`</s>`, so a Qwen instruct model that ends its turn with
/// `<|im_end|>` would run on until `max_new_tokens`.
pub fn resolve_stop_ids(dir: impl AsRef<Path>, cfg: &ModelConfig, tok: &Tokenizer) -> Vec<u32> {
    let mut ids = cfg.eos_token_ids.clone();
    if let Ok(s) = fs::read_to_string(dir.as_ref().join("generation_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            ids.extend(id_list(v.get("eos_token_id")));
        }
    }
    if ids.is_empty() {
        ids.extend(STOP_TOKEN_NAMES.iter().filter_map(|n| tok.token_to_id(n)));
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Incremental detokenizer for streaming output.
///
/// Decoding one token at a time corrupts multi-byte characters and drops leading spaces, and
/// re-decoding the whole sequence each step is quadratic. This keeps a small sliding window
/// (the "prefix/read offset" scheme used by text-generation-inference): it emits text only
/// once the window decodes to something longer than before and does not end in U+FFFD
/// (an incomplete UTF-8 sequence).
pub struct TokenStream<'a> {
    tok: &'a Tokenizer,
    ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl<'a> TokenStream<'a> {
    pub fn new(tok: &'a Tokenizer) -> Self {
        Self { tok, ids: Vec::new(), prefix_offset: 0, read_offset: 0 }
    }

    /// Seeds the window with the tail of the prompt so the first generated token is decoded
    /// with its real left context (otherwise its leading space is lost).
    pub fn with_context(mut self, prompt_ids: &[u32]) -> Self {
        let tail = &prompt_ids[prompt_ids.len().saturating_sub(6)..];
        self.ids = tail.to_vec();
        self.prefix_offset = 0;
        self.read_offset = tail.len();
        self
    }

    pub fn push(&mut self, id: u32) -> Result<Option<String>> {
        self.ids.push(id);
        self.emit(false)
    }

    /// Emits whatever is still buffered (e.g. a trailing incomplete character) at end of stream.
    pub fn flush(&mut self) -> Result<Option<String>> {
        self.emit(true)
    }

    fn emit(&mut self, force: bool) -> Result<Option<String>> {
        let prefix = self.tok.decode(&self.ids[self.prefix_offset..self.read_offset], true)?;
        let text = self.tok.decode(&self.ids[self.prefix_offset..], true)?;
        if text.len() > prefix.len() && (force || !text.ends_with('\u{FFFD}')) {
            let out = text.get(prefix.len()..).map(str::to_owned);
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            Ok(out)
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    /// A tiny WordLevel tokenizer (`<|endoftext|>`=0, `<|im_start|>`=1, `<|im_end|>`=2,
    /// `w0..w39`=3..42) so tests and fixtures need no network or external files.
    pub fn word_level_json() -> String {
        let mut vocab = String::from(r#""<|endoftext|>":0,"<|im_start|>":1,"<|im_end|>":2,"[UNK]":43"#);
        for i in 0..40 {
            vocab.push_str(&format!(r#","w{i}":{}"#, i + 3));
        }
        format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[
              {{"id":0,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}},
              {{"id":1,"content":"<|im_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}},
              {{"id":2,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}],
              "normalizer":null,"pre_tokenizer":{{"type":"WhitespaceSplit"}},"post_processor":null,
              "decoder":null,"model":{{"type":"WordLevel","vocab":{{{vocab}}},"unk_token":"[UNK]"}}}}"#
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::word_level_json;
    use super::*;

    #[test]
    fn encode_decode_roundtrip_and_specials() {
        let tok = Tokenizer::from_json_str(&word_level_json()).unwrap();
        let ids = tok.encode("w3 w7 <|im_end|>", false).unwrap();
        assert_eq!(ids, vec![6, 10, 2]);
        assert_eq!(tok.decode(&ids, false).unwrap(), "w3 w7 <|im_end|>");
        assert_eq!(tok.decode(&ids, true).unwrap(), "w3 w7");
        assert_eq!(tok.token_to_id("<|im_end|>"), Some(2));
    }

    #[test]
    fn stream_reassembles_the_same_text_as_a_full_decode() {
        let tok = Tokenizer::from_json_str(&word_level_json()).unwrap();
        let ids = [6u32, 10, 4, 22];
        let mut s = TokenStream::new(&tok);
        let mut out = String::new();
        for id in ids {
            if let Some(t) = s.push(id).unwrap() {
                out.push_str(&t);
            }
        }
        if let Some(t) = s.flush().unwrap() {
            out.push_str(&t);
        }
        assert_eq!(out, tok.decode(&ids, true).unwrap());
    }

    #[test]
    fn seeded_stream_keeps_the_leading_space_of_the_first_token() {
        let tok = Tokenizer::from_json_str(&word_level_json()).unwrap();
        let mut s = TokenStream::new(&tok).with_context(&[6, 10]);
        assert_eq!(s.push(4).unwrap().as_deref(), Some(" w1"));
    }

    #[test]
    fn stop_ids_union_config_and_generation_config_else_fallback_names() {
        let tok = Tokenizer::from_json_str(&word_level_json()).unwrap();
        let dir = std::env::temp_dir().join(format!("burn_llm_stop_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut cfg = ModelConfig::from_json_str(
            r#"{"vocab_size":50,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,
                "num_attention_heads":2,"eos_token_id":0}"#,
        )
        .unwrap();

        // config only
        assert_eq!(resolve_stop_ids(&dir, &cfg, &tok), vec![0]);

        // + generation_config (list) => union, sorted, deduped
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id":[2,0]}"#).unwrap();
        assert_eq!(resolve_stop_ids(&dir, &cfg, &tok), vec![0, 2]);

        // neither names one => fall back to well-known names present in the vocab
        cfg.eos_token_ids.clear();
        std::fs::remove_file(dir.join("generation_config.json")).unwrap();
        assert_eq!(resolve_stop_ids(&dir, &cfg, &tok), vec![0, 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
