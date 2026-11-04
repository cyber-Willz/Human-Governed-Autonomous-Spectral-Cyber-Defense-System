use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use burn_ndarray::{NdArray, NdArrayDevice};
use rand::rngs::StdRng;
use rand::SeedableRng;

use burn_llm::chat::{self, ChatOptions, Message};
use burn_llm::config::ModelConfig;
use burn_llm::diagnostics;
use burn_llm::generate::{generate, FinishReason, SamplingConfig};
use burn_llm::tokenizer::{load_tokenizer, resolve_stop_ids, TokenStream};
use burn_llm::weights::load_model;

type B = NdArray<f32>;

const USAGE: &str = "usage: generate <model_dir> [options] <prompt>
  --chat                 wrap the prompt in the model's chat template (ChatML / Llama-3)
  --system <text>        system message (with --chat)
  --no-think             Qwen3: disable the thinking block (with --chat)
  --max-new <n>          tokens to generate            (default 128)
  --temp <f>             temperature, 0 = greedy       (default 0.7)
  --top-k <n> | --top-p <f> | --min-p <f> | --rep-penalty <f>
  --seed <n>             RNG seed                      (default 0)
  --prefill-chunk <n>    prompt tokens per forward     (default 512)
  --check                run a numerical health probe on the prompt and exit";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir: PathBuf = args.next().context(USAGE)?.into();

    let mut cfg = SamplingConfig { max_new_tokens: 128, ..Default::default() };
    let (mut use_chat, mut check, mut no_think) = (false, false, false);
    let (mut system, mut seed) = (None::<String>, 0u64);
    let mut prompt_parts: Vec<String> = Vec::new();

    fn next_val(a: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
        a.next().with_context(|| format!("{flag} needs a value\n{USAGE}"))
    }
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chat" => use_chat = true,
            "--no-think" => no_think = true,
            "--check" => check = true,
            "--system" => system = Some(next_val(&mut args, &a)?),
            "--max-new" => cfg.max_new_tokens = next_val(&mut args, &a)?.parse()?,
            "--temp" => cfg.temperature = next_val(&mut args, &a)?.parse()?,
            "--top-k" => cfg.top_k = Some(next_val(&mut args, &a)?.parse()?),
            "--top-p" => cfg.top_p = Some(next_val(&mut args, &a)?.parse()?),
            "--min-p" => cfg.min_p = Some(next_val(&mut args, &a)?.parse()?),
            "--rep-penalty" => cfg.repetition_penalty = next_val(&mut args, &a)?.parse()?,
            "--seed" => seed = next_val(&mut args, &a)?.parse()?,
            "--prefill-chunk" => cfg.prefill_chunk = next_val(&mut args, &a)?.parse()?,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            s if s.starts_with("--") => bail!("unknown option {s}\n{USAGE}"),
            _ => prompt_parts.push(a),
        }
    }
    let prompt = prompt_parts.join(" ");
    if prompt.is_empty() {
        bail!("{USAGE}");
    }

    let device = NdArrayDevice::default();
    let model_cfg = ModelConfig::from_json_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let tok = load_tokenizer(&dir)?;
    let stop_ids = resolve_stop_ids(&dir, &model_cfg, &tok);

    let t0 = Instant::now();
    let model = load_model::<B>(&dir, None, &device)?;
    eprintln!("loaded {} layers in {:.1}s; stop ids {:?}", model.num_layers(), t0.elapsed().as_secs_f32(), stop_ids);

    let (text, add_special) = if use_chat {
        let fmt = chat::detect_format(&tok)
            .context("tokenizer has neither ChatML nor Llama-3 special tokens; omit --chat")?;
        let mut msgs = Vec::new();
        if let Some(s) = &system {
            msgs.push(Message::new("system", s));
        }
        msgs.push(Message::new("user", &prompt));
        (chat::render(fmt, &msgs, &ChatOptions { no_think }), false)
    } else {
        (prompt.clone(), true)
    };
    let ids = tok.encode(&text, add_special)?;

    if check {
        let r = diagnostics::probe(&model, &ids, &device);
        for l in &r.layers {
            println!("layer {:>3}  rms {:>10.4}  finite {}", l.layer, l.rms, l.all_finite);
        }
        println!("logits finite: {}  => {}", r.logits_finite, if r.healthy() { "OK" } else { "UNHEALTHY" });
        return Ok(());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut stream = TokenStream::new(&tok).with_context(&ids);
    print!("{}", if use_chat { "" } else { &prompt });
    let t1 = Instant::now();
    let out = generate(&model, &ids, &stop_ids, &cfg, &device, &mut rng, |id| {
        if let Ok(Some(piece)) = stream.push(id) {
            print!("{piece}");
            let _ = std::io::stdout().flush();
        }
    })?;
    if let Ok(Some(rest)) = stream.flush() {
        print!("{rest}");
    }
    println!();
    let secs = t1.elapsed().as_secs_f32();
    eprintln!(
        "[{} prompt tok, {} new tok, {:?}, {:.2} tok/s]",
        ids.len(),
        out.new_tokens,
        match out.finish { FinishReason::Stop => "stop", FinishReason::Length => "length" },
        out.new_tokens as f32 / secs.max(1e-9)
    );
    Ok(())
}
