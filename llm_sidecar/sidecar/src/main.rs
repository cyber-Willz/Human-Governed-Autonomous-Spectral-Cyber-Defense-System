//! `llm_sidecar <model_dir>` -- serves completions over stdin/stdout.
//!
//! Framing (binary-safe, no JSON dependency):
//!   request : `REQ <max_new_tokens> <prompt_byte_len>\n<prompt bytes>`
//!   reply   : `RES <byte_len>\n<bytes>`  or  `ERR <message without newlines>\n`
//! Diagnostics go to stderr only. EOF on stdin ends the process.
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use burn_llm::chat::{self, ChatOptions, Message};
use burn_llm::config::ModelConfig;
use burn_llm::generate::{generate, SamplingConfig};
use burn_llm::tokenizer::{load_tokenizer, resolve_stop_ids};
use burn_llm::weights::load_model;
use burn_ndarray::{NdArray, NdArrayDevice};
use rand::rngs::StdRng;
use rand::SeedableRng;

type B = NdArray<f32>;
const MAX_PROMPT_BYTES: usize = 64 * 1024;

fn main() -> Result<()> {
    let dir: PathBuf = std::env::args().nth(1).context("usage: llm_sidecar <model_dir>")?.into();
    let device = NdArrayDevice::default();
    let cfg = ModelConfig::from_json_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let tok = load_tokenizer(&dir)?;
    let stop = resolve_stop_ids(&dir, &cfg, &tok);
    let model = load_model::<B>(&dir, None, &device)?;
    let fmt = chat::detect_format(&tok);
    let max_pos = model.config.max_position_embeddings;
    eprintln!("llm_sidecar: ready ({} layers, chat={})", model.num_layers(), fmt.is_some());

    let mut input = BufReader::new(std::io::stdin().lock());
    let mut out = std::io::stdout().lock();
    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            return Ok(());
        }
        let reply: Result<String> = (|| {
            let mut it = line.split_whitespace();
            if it.next() != Some("REQ") {
                bail!("bad request line");
            }
            let max_new: usize = it.next().context("max_new")?.parse()?;
            let len: usize = it.next().context("len")?.parse()?;
            if len > MAX_PROMPT_BYTES {
                // drain so the stream stays in sync
                std::io::copy(&mut (&mut input).take(len as u64), &mut std::io::sink())?;
                bail!("prompt too large");
            }
            let mut buf = vec![0u8; len];
            input.read_exact(&mut buf)?;
            let prompt = String::from_utf8_lossy(&buf).into_owned();
            let (text, special) = match fmt {
                Some(f) => (chat::render(f, &[Message::new("user", &prompt)], &ChatOptions { no_think: true }), false),
                None => (prompt, true),
            };
            let ids = tok.encode(&text, special)?;
            if ids.len() + 1 >= max_pos {
                bail!("prompt exceeds model context");
            }
            let scfg = SamplingConfig { max_new_tokens: max_new.min(256), ..SamplingConfig::greedy() };
            let mut rng = StdRng::seed_from_u64(0);
            let mut new_ids = Vec::new();
            generate(&model, &ids, &stop, &scfg, &device, &mut rng, |i| new_ids.push(i))?;
            Ok(tok.decode(&new_ids, true)?)
        })();
        match reply {
            Ok(s) => {
                write!(out, "RES {}\n", s.len())?;
                out.write_all(s.as_bytes())?;
            }
            Err(e) => writeln!(out, "ERR {}", e.to_string().replace('\n', " "))?,
        }
        out.flush()?;
    }
}
