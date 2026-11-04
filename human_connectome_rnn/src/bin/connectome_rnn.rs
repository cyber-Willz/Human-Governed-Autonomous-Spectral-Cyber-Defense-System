use hcrnn::network::{default_roles, default_threads};
use hcrnn::topology::{harmonics, spectral_modules, topology_report};
use hcrnn::train::{delay_cue_variability_by_system, evaluate, harmonic_profile, make_trials, train};
use hcrnn::{Calibration, Connectome, ConnectomeLstm, ConnectomeRnn, LstmConfig, MaskMode, MemoryGuidedTask, RnnConfig, SequenceModel, TrainConfig};
use std::collections::{HashMap, HashSet};
use std::path::Path;

const HELP: &str = "\
connectome_rnn - recurrent neural networks constrained by a human structural connectome

USAGE:
  connectome_rnn <inspect|train|compare|bench> [options]

DATA (pick one; default is --synthetic 40):
  --tvb DIR              TVB-style folder (weights.txt, tract_lengths.txt, centres.txt),
                         e.g. the human 66-region connectome (see scripts/fetch_tvb66.sh)
  --csv FILE             N x N weight matrix; with --lengths FILE (mm) and --labels FILE
                         (rows: name[,hemi[,system[,x,y,z]]])
  --features FILE        parcels x features table (no tractography needed); needs --labels with
                         x,y,z positions. Graph = correlation kNN of feature profiles
                         (--knn K, default 10); lengths from positions (--max-length-mm M to
                         rescale the longest to M mm)
  --synthetic N          synthetic human-like connectome, N regions per hemisphere
  --density D            target density for --synthetic (default 0.25)
  --data-seed S          seed for --synthetic (default 1)
  --linear-weights       use raw weights (default: log-compress heavy-tailed weights)

MODEL:
  --model M              rnn (leaky rate network, default) | lstm (gated cells on the same wiring)
  --gate-scale X         lstm: initial i/f/o recurrent strength vs the candidate path (default 0.5)
  --forget-bias X        lstm: initial forget-gate bias (default 2.0; tuned, see README)
  --mode M               connectome | degree-preserving | density-matched | dense  (train)
  --no-delays            ignore conduction delays
  --velocity V           conduction velocity in mm/ms = m/s (default 5)
  --dt MS --tau MS       Euler step and membrane time constant (default 5, 20)
  --calibration C        power | arnoldi   initial-strength calibration (default power)
  --target-radius R      Arnoldi: radius of the delayed linearised map (default 0.90)
  --gain G               power: radius of W (default 0.9)
  --inter-gain G         inter-regional vs local strength (default 1.0)
  --input-regions LIST   comma-separated indices or names (default: visual regions)
  --output-regions LIST  comma-separated indices or names (default: somatomotor regions)

TASK:
  --delay-min N --delay-max N   memory delay in steps (default 20, 40); longer = harder

TRAINING:
  --steps N (400)  --batch N (24)  --lr X (0.006)  --clip X (1.0)  --seed N (1)
  --log-every N (50)  --threads N  --out DIR (save params + weights)
  --energy-lambda X      connectome-Laplacian Dirichlet-energy regulariser (default 0)
  --energy-start N       first step at which the regulariser is active (default 0)
  --radius-cap X         project weights so the Arnoldi radius stays <= X (default off)
  --cosine               cosine learning-rate decay;  --warmup N  linear warm-up steps
  --seeds N              compare/bench: number of seeds (default 4)
  --seed0 N              compare: first seed (default 1); use other values to keep tuning and evaluation seeds apart
  --modes LIST           compare: comma-separated subset of wiring modes (default: all)
";

struct Args {
    cmd: String,
    kv: HashMap<String, String>,
    flags: HashSet<String>,
}

const FLAGS: &[&str] = &["no-delays", "linear-weights", "help", "cosine"];

impl Args {
    fn parse() -> Result<Args, String> {
        let mut cmd = String::new();
        let mut kv = HashMap::new();
        let mut flags = HashSet::new();
        let mut it = std::env::args().skip(1).peekable();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                if FLAGS.contains(&name) {
                    flags.insert(name.to_string());
                } else {
                    let v = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
                    kv.insert(name.to_string(), v);
                }
            } else if cmd.is_empty() {
                cmd = a;
            } else {
                return Err(format!("unexpected argument '{a}'"));
            }
        }
        Ok(Args { cmd, kv, flags })
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.contains(k)
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        match self.get(k) {
            Some(v) => v.parse::<T>().map_err(|_| format!("--{k}: cannot parse '{v}'")),
            None => Ok(default),
        }
    }
}

fn load_connectome(a: &Args) -> Result<Connectome, String> {
    let c = if let Some(d) = a.get("tvb") {
        Connectome::from_tvb_dir(Path::new(d))?
    } else if let Some(f) = a.get("features") {
        let labels = a.get("labels").ok_or("--features needs --labels (name,hemi,system,x,y,z)")?;
        let max_len = match a.get("max-length-mm") {
            Some(v) => Some(v.parse::<f64>().map_err(|_| format!("--max-length-mm: cannot parse '{v}'"))?),
            None => None,
        };
        Connectome::from_features(Path::new(f), Path::new(labels), a.num("knn", 10usize)?, max_len)?
    } else if let Some(w) = a.get("csv") {
        Connectome::from_csv(Path::new(w), a.get("lengths").map(Path::new), a.get("labels").map(Path::new))?
    } else {
        Connectome::synthetic_human(a.num("synthetic", 40usize)?, a.num("density", 0.25f64)?, a.num("data-seed", 1u64)?)?
    };
    Ok(if a.flag("linear-weights") { c } else { c.log_compressed() })
}

fn region_list(s: &str, conn: &Connectome) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for tok in s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()) {
        if let Ok(i) = tok.parse::<usize>() {
            if i >= conn.n() {
                return Err(format!("region index {i} out of range"));
            }
            out.push(i);
        } else if let Some(i) = conn.regions.iter().position(|r| r.name == tok) {
            out.push(i);
        } else {
            return Err(format!("unknown region '{tok}'"));
        }
    }
    Ok(out)
}

fn build_config(a: &Args, conn: &Connectome, mode: MaskMode) -> Result<RnnConfig, String> {
    let (di, do_) = match (a.get("input-regions"), a.get("output-regions")) {
        (Some(_), Some(_)) => (Vec::new(), Vec::new()),
        _ => default_roles(conn)?,
    };
    let input_regions = match a.get("input-regions") {
        Some(s) => region_list(s, conn)?,
        None => di,
    };
    let output_regions = match a.get("output-regions") {
        Some(s) => region_list(s, conn)?,
        None => do_,
    };
    let d = RnnConfig::default();
    let calibration = match a.get("calibration").unwrap_or("power") {
        "arnoldi" => Calibration::Arnoldi,
        "power" => Calibration::Power,
        other => return Err(format!("unknown --calibration '{other}'")),
    };
    Ok(RnnConfig {
        mode,
        dt_ms: a.num("dt", d.dt_ms)?,
        tau_ms: a.num("tau", d.tau_ms)?,
        velocity_mm_per_ms: a.num("velocity", d.velocity_mm_per_ms)?,
        use_delays: !a.flag("no-delays"),
        calibration,
        gain: a.num("gain", d.gain)?,
        target_radius: a.num("target-radius", d.target_radius)?,
        inter_gain: a.num("inter-gain", d.inter_gain)?,
        n_in: MemoryGuidedTask::N_IN,
        n_out: MemoryGuidedTask::N_OUT,
        input_regions,
        output_regions,
        seed: a.num("seed", 1u64)?,
        ..d
    })
}

#[derive(Clone)]
enum ModelKind {
    Rnn,
    Lstm(LstmConfig),
}

fn model_kind(a: &Args) -> Result<ModelKind, String> {
    match a.get("model").unwrap_or("rnn") {
        "rnn" => Ok(ModelKind::Rnn),
        "lstm" => {
            let d = LstmConfig::default();
            Ok(ModelKind::Lstm(LstmConfig { gate_scale: a.num("gate-scale", d.gate_scale)?, forget_bias: a.num("forget-bias", d.forget_bias)? }))
        }
        other => Err(format!("unknown --model '{other}' (rnn or lstm)")),
    }
}

fn task_from_args(a: &Args) -> Result<MemoryGuidedTask, String> {
    let d = MemoryGuidedTask::default();
    let t = MemoryGuidedTask { delay_min: a.num("delay-min", d.delay_min)?, delay_max: a.num("delay-max", d.delay_max)?, ..d };
    if t.delay_max < t.delay_min {
        return Err("--delay-max must be >= --delay-min".into());
    }
    Ok(t)
}

fn train_config(a: &Args) -> Result<TrainConfig, String> {
    let d = TrainConfig::default();
    Ok(TrainConfig {
        steps: a.num("steps", d.steps)?,
        batch: a.num("batch", d.batch)?,
        lr: a.num("lr", d.lr)?,
        clip: a.num("clip", d.clip)?,
        seed: a.num("seed", d.seed)?,
        log_every: a.num("log-every", d.log_every)?,
        eval_trials: d.eval_trials,
        threads: a.num("threads", default_threads())?,
        energy_lambda: a.num("energy-lambda", d.energy_lambda)?,
        energy_start: a.num("energy-start", d.energy_start)?,
        radius_cap: match a.get("radius-cap") {
            Some(v) => Some(v.parse::<f64>().map_err(|_| format!("--radius-cap: cannot parse '{v}'"))?),
            None => None,
        },
        project_every: d.project_every,
        warmup: a.num("warmup", d.warmup)?,
        cosine: a.flag("cosine"),
    })
}

fn names(conn: &Connectome, idx: &[usize]) -> String {
    idx.iter().map(|&i| conn.regions[i].name.as_str()).collect::<Vec<_>>().join(", ")
}

fn describe_net(net: &ConnectomeRnn) -> String {
    let inter = net.edges.iter().filter(|e| net.region_of(e.post) != net.region_of(e.pre)).count();
    format!(
        "units: {} ({} E + {} I)   synapses: {} ({} inter-regional)   trainable params: {}   max delay: {} steps ({:.0} ms)\nradius of W: {:.3}   radius of delayed dynamics (Arnoldi): {:.3}",
        net.n_units,
        net.n_regions,
        net.n_regions,
        net.edges.len(),
        inter,
        net.num_params(),
        net.max_delay(),
        net.max_delay() as f64 * net.cfg.dt_ms,
        net.weight_radius(),
        net.spectral_radius()
    )
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len().max(1) as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = if v.len() > 1 { v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    (m, var.sqrt())
}

/// One trained run's headline numbers.
struct RunResult {
    err: f64,
    steps_to_30: Option<usize>,
    radius: f64,
    low_frac: f64,
    train_seconds: f64,
}

fn one_run(
    conn: &Connectome,
    cfg: RnnConfig,
    kind: &ModelKind,
    tcfg: &TrainConfig,
    task: &MemoryGuidedTask,
    harm: &hcrnn::topology::Harmonics,
) -> Result<RunResult, String> {
    match kind {
        ModelKind::Rnn => {
            let mut net = ConnectomeRnn::new(conn, cfg)?;
            let mut r = one_run_model(&mut net, tcfg, task, harm)?;
            r.radius = net.spectral_radius();
            Ok(r)
        }
        ModelKind::Lstm(l) => {
            let mut net = ConnectomeLstm::new(conn, cfg, l.clone())?;
            one_run_model(&mut net, tcfg, task, harm)
        }
    }
}

fn one_run_model<M: SequenceModel>(
    net: &mut M,
    tcfg: &TrainConfig,
    task: &MemoryGuidedTask,
    harm: &hcrnn::topology::Harmonics,
) -> Result<RunResult, String> {
    net.check_constraints()?;
    let rep = train(net, task, tcfg, |_| {});
    net.check_constraints()?;
    let test = make_trials(task, 256, 0xC0FFEE);
    let m = evaluate(net, &test);
    let steps_to_30 = rep.log.iter().find(|l| l.eval.angular_error_deg < 30.0).map(|l| l.step);
    let hp = harmonic_profile(net, &test[..96], harm);
    Ok(RunResult { err: m.angular_error_deg, steps_to_30, radius: f64::NAN, low_frac: hp.low, train_seconds: rep.seconds })
}

/// Training log, held-out evaluation and activity analysis shared by both models.
fn train_and_report<M: SequenceModel>(
    net: &mut M,
    conn: &Connectome,
    tcfg: &TrainConfig,
    task: &MemoryGuidedTask,
    out: Option<&str>,
) -> Result<(), String> {
    println!("\n step   train-loss   eval-loss   ang.err(deg)   acc<30deg   |grad|   radius   sec");
    let rep = train(net, task, tcfg, |l| {
        let radius = if l.radius.is_nan() { "     -".to_string() } else { format!("{:6.3}", l.radius) };
        println!(
            "{:5}   {:10.4}   {:9.4}   {:12.1}   {:9.2}   {:6.3}   {}   {:5.1}",
            l.step, l.train_loss, l.eval.loss, l.eval.angular_error_deg, l.eval.acc_30deg, l.grad_norm, radius, l.seconds
        );
    });
    net.check_constraints()?;
    println!(
        "\nfinal: loss {:.4}  angular error {:.1} deg (chance 90)  within 30 deg: {:.0}%  [{:.1}s]  constraints verified",
        rep.final_eval.loss, rep.final_eval.angular_error_deg, rep.final_eval.acc_30deg * 100.0, rep.seconds
    );
    let test = make_trials(task, 256, 0xC0FFEE);
    let m = evaluate(net, &test);
    println!("held-out (256 new trials): loss {:.4}  angular error {:.1} deg  within 30 deg: {:.0}%", m.loss, m.angular_error_deg, m.acc_30deg * 100.0);
    let hp = harmonic_profile(net, &test[..96], &harmonics(conn));
    println!(
        "\nCue-dependent delay activity on connectome harmonics: low {:.0}%  mid {:.0}%  high {:.0}%  (Dirichlet quotient {:.3}; 0 = smooth, 2 = alternating)",
        hp.low * 100.0, hp.mid * 100.0, hp.high * 100.0, hp.dirichlet_quotient
    );
    println!("Cue-dependent variability of E rates at end of delay, by system:");
    for (s, n, v) in delay_cue_variability_by_system(net, &test[..64]) {
        println!("  {:<18} {:3} regions   {:.4}", s, n, v);
    }
    if let Some(dir) = out {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let what = net.save_to_dir(Path::new(dir))?;
        println!("\nsaved {what} to {dir}");
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let a = Args::parse()?;
    if a.flag("help") || a.cmd.is_empty() || a.cmd == "help" {
        print!("{HELP}");
        return Ok(());
    }
    let conn = load_connectome(&a)?;
    match a.cmd.as_str() {
        "inspect" => {
            println!("{}", conn.summary());
            let (inp, out) = default_roles(&conn).unwrap_or_default();
            if !inp.is_empty() {
                println!("default input regions : {}", names(&conn, &inp));
                println!("default output regions: {}", names(&conn, &out));
            }
            let t = topology_report(&conn);
            println!(
                "\ntopology (nbsc + krylov_ds): non-backtracking radius rho_B = {:.3}   ||A||_2 = {:.3}   ||A||/rho_B = {:.3} ({})   Fiedler value = {:.4}",
                t.rho_b,
                t.adj_norm,
                t.expansion_ratio,
                if t.expansion_ratio > 1.0 { "A/rho_B is expansive" } else { "A/rho_B is non-expansive" },
                t.fiedler
            );
            let h = harmonics(&conn);
            let lows: Vec<String> = h.values.iter().take(6).map(|v| format!("{v:.3}")).collect();
            println!("lowest Laplacian harmonics (eigenvalues): {}", lows.join(", "));
            let k = a.num("modules", 4usize)?;
            let assign = spectral_modules(&conn, k, 1)?;
            println!("\nspectral modules (spectral_hypergraph, k={k}):");
            for m in 0..k {
                let members: Vec<usize> = (0..conn.n()).filter(|&i| assign[i] == m).collect();
                if members.is_empty() {
                    continue;
                }
                let l = members.iter().filter(|&&i| conn.regions[i].hemi == 'L').count();
                let r = members.iter().filter(|&&i| conn.regions[i].hemi == 'R').count();
                let mut sys: HashMap<&str, usize> = HashMap::new();
                for &i in &members {
                    *sys.entry(conn.regions[i].system.as_str()).or_default() += 1;
                }
                let mut sv: Vec<_> = sys.into_iter().collect();
                sv.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(y.0)));
                let top: Vec<String> = sv.iter().take(3).map(|(s, c)| format!("{s}:{c}")).collect();
                println!("  module {m}: {:3} regions   L/R {l}/{r}   systems {}", members.len(), top.join(" "));
            }
            Ok(())
        }
        "train" => {
            let mode = match a.get("mode") {
                Some(m) => MaskMode::parse(m).ok_or_else(|| format!("unknown --mode '{m}'"))?,
                None => MaskMode::Connectome,
            };
            let cfg = build_config(&a, &conn, mode)?;
            let tcfg = train_config(&a)?;
            let kind = model_kind(&a)?;
            let task = task_from_args(&a)?;
            println!("{}", conn.summary());
            println!("input regions : {}", names(&conn, &cfg.input_regions));
            println!("output regions: {}", names(&conn, &cfg.output_regions));
            println!("task: memory delay {}..{} steps", task.delay_min, task.delay_max);
            match kind {
                ModelKind::Rnn => {
                    let mut net = ConnectomeRnn::new(&conn, cfg)?;
                    net.check_constraints()?;
                    println!("model: rnn   mode: {}\n{}", mode.name(), describe_net(&net));
                    train_and_report(&mut net, &conn, &tcfg, &task, a.get("out"))
                }
                ModelKind::Lstm(l) => {
                    let mut net = ConnectomeLstm::new(&conn, cfg, l)?;
                    net.check_constraints()?;
                    println!(
                        "model: lstm (gate-scale {}, forget-bias {})   mode: {}\nunits: {} ({} E + {} I)   synapses per gate: {}   trainable params: {}   max delay: {} steps ({:.0} ms)",
                        net.lcfg.gate_scale, net.lcfg.forget_bias, mode.name(), net.base.n_units, net.base.n_regions, net.base.n_regions,
                        net.base.edges.len(), net.num_params(), net.base.max_delay(), net.base.max_delay() as f64 * net.base.cfg.dt_ms
                    );
                    train_and_report(&mut net, &conn, &tcfg, &task, a.get("out"))
                }
            }
        }
        "compare" => {
            let tcfg0 = train_config(&a)?;
            let task = task_from_args(&a)?;
            let kind = model_kind(&a)?;
            let seeds: u64 = a.num("seeds", 4u64)?;
            let seed0: u64 = a.num("seed0", 1u64)?;
            let harm = harmonics(&conn);
            println!("{}\n", conn.summary());
            println!("{:<18} {:>8} {:>8} {:>7} {:>13} {:>11} {:>9} {:>8}", "wiring", "rho_B", "|A|/rho_B", "fiedler", "err(deg) m+-sd", "failures", "low-harm", "sec/run");
            let wanted: Option<Vec<String>> = a.get("modes").map(|m| m.split(',').map(|t| t.trim().to_string()).collect());
            if let Some(w) = &wanted {
                for t in w {
                    if !MaskMode::ALL.iter().any(|m| m.name() == t) {
                        return Err(format!("--modes: unknown mode '{t}'"));
                    }
                }
            }
            for mode in MaskMode::ALL {
                if let Some(w) = &wanted {
                    if !w.iter().any(|t| t == mode.name()) {
                        continue;
                    }
                }
                // topology of the graph this mode actually wires
                let probe = ConnectomeRnn::new(&conn, RnnConfig { seed: 1, ..build_config(&a, &conn, mode)? })?;
                let wg = wiring_connectome(&probe, &conn);
                let t = topology_report(&wg);
                let (mut errs, mut lows, mut secs, mut fails) = (Vec::new(), Vec::new(), Vec::new(), 0);
                for s in seed0..seed0 + seeds {
                    let mut cfg = build_config(&a, &conn, mode)?;
                    cfg.seed = s;
                    let mut tcfg = tcfg0.clone();
                    tcfg.seed = s;
                    let r = one_run(&conn, cfg, &kind, &tcfg, &task, &harm)?;
                    if r.err > 45.0 {
                        fails += 1;
                    }
                    errs.push(r.err);
                    lows.push(r.low_frac);
                    secs.push(r.train_seconds);
                    eprintln!("[{} seed {s}] err {:.1}", mode.name(), r.err);
                }
                let (m, sd) = mean_sd(&errs);
                let (lm, _) = mean_sd(&lows);
                println!(
                    "{:<18} {:>8.2} {:>8.3} {:>7.3} {:>6.1} +- {:<5.1} {:>8}/{} {:>8.0}% {:>8.1}",
                    mode.name(), t.rho_b, t.expansion_ratio, t.fiedler, m, sd, fails, seeds, lm * 100.0, mean_sd(&secs).0
                );
            }
            println!("\n(held-out 256 trials per run, {seeds} seeds, chance = 90 deg, failure = error > 45 deg; low-harm = share of cue-dependent delay variance in the lowest third of connectome harmonics)");
            Ok(())
        }
        "bench" => {
            let tcfg0 = train_config(&a)?;
            let task = task_from_args(&a)?;
            let seeds: u64 = a.num("seeds", 4u64)?;
            let harm = harmonics(&conn);
            struct V {
                name: &'static str,
                calib: Calibration,
                radius: f64,
                energy: f64,
                start: usize,
                cap: Option<f64>,
                cosine: bool,
                lr: f64,
            }
            let p = Calibration::Power;
            let ar = Calibration::Arnoldi;
            let variants = [
                V { name: "v0.1 baseline (power, gain .9)", calib: p, radius: 0.0, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "arnoldi r=0.95", calib: ar, radius: 0.95, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "arnoldi r=0.975", calib: ar, radius: 0.975, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "arnoldi r=1.0", calib: ar, radius: 1.0, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "arnoldi r=1.03", calib: ar, radius: 1.03, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "r=0.975 + energy 0.05", calib: ar, radius: 0.975, energy: 0.05, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "r=0.975 + energy 0.3", calib: ar, radius: 0.975, energy: 0.3, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "r=0.975 + radius cap 1.05", calib: ar, radius: 0.975, energy: 0.0, start: 0, cap: Some(1.05), cosine: false, lr: 0.0 },
                V { name: "r=0.975 + cosine lr", calib: ar, radius: 0.975, energy: 0.0, start: 0, cap: None, cosine: true, lr: 0.0 },
                V { name: "R2: arnoldi r=0.85", calib: ar, radius: 0.85, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "R2: arnoldi r=0.90", calib: ar, radius: 0.90, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "R2: r=0.90 + energy .005", calib: ar, radius: 0.90, energy: 0.005, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "R2: r=0.90 + energy .02 @150", calib: ar, radius: 0.90, energy: 0.02, start: 150, cap: None, cosine: false, lr: 0.0 },
                V { name: "R2: r=0.90 + energy .1 @200", calib: ar, radius: 0.90, energy: 0.1, start: 200, cap: None, cosine: false, lr: 0.0 },
                V { name: "R3: baseline lr .006", calib: p, radius: 0.0, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.006 },
                V { name: "R3: arnoldi r=0.95", calib: ar, radius: 0.95, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "R3: arnoldi r=0.90", calib: ar, radius: 0.90, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.0 },
                V { name: "R3: arnoldi r=0.90 lr .006", calib: ar, radius: 0.90, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.006 },
                V { name: "R4: baseline lr .01", calib: p, radius: 0.0, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.01 },
                V { name: "R4: baseline lr .015", calib: p, radius: 0.0, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.015 },
                V { name: "R4: arnoldi r=0.90 lr .01", calib: ar, radius: 0.90, energy: 0.0, start: 0, cap: None, cosine: false, lr: 0.01 },
            ];
            let only = a.get("only");
            println!("{}\n{} seeds, {} steps each, wiring = connectome\n", conn.summary(), seeds, tcfg0.steps);
            println!("{:<32} {:>13} {:>10} {:>12} {:>9} {:>9} {:>8}", "variant", "err(deg) m+-sd", "failures", "steps->30deg", "radius", "low-harm", "wins");
            let mut base: Vec<f64> = Vec::new();
            for (vi, v) in variants.iter().enumerate() {
                if let Some(o) = only {
                    if !v.name.contains(o) && vi != 0 {
                        continue;
                    }
                }
                let (mut errs, mut st, mut rad, mut low, mut fails) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0);
                for s in 1..=seeds {
                    let mut cfg = build_config(&a, &conn, MaskMode::Connectome)?;
                    cfg.calibration = v.calib;
                    if v.radius > 0.0 {
                        cfg.target_radius = v.radius;
                    }
                    cfg.seed = s;
                    let mut tcfg = tcfg0.clone();
                    tcfg.seed = s;
                    tcfg.energy_lambda = v.energy;
                    tcfg.energy_start = v.start;
                    tcfg.radius_cap = v.cap;
                    tcfg.cosine = v.cosine;
                    if v.lr > 0.0 {
                        tcfg.lr = v.lr;
                    }
                    tcfg.log_every = 25;
                    let r = one_run(&conn, cfg, &ModelKind::Rnn, &tcfg, &task, &harm)?;
                    if r.err > 45.0 {
                        fails += 1;
                    }
                    errs.push(r.err);
                    st.push(r.steps_to_30.map(|x| x as f64).unwrap_or(tcfg.steps as f64 * 1.25));
                    rad.push(r.radius);
                    low.push(r.low_frac);
                }
                if vi == 0 {
                    base = errs.clone();
                }
                let wins = errs.iter().zip(&base).filter(|(e, b)| e < b).count();
                let (m, sd) = mean_sd(&errs);
                println!(
                    "{:<32} {:>6.1} +- {:<5.1} {:>8}/{} {:>12.0} {:>9.3} {:>8.0}% {:>6}/{}",
                    v.name, m, sd, fails, seeds, mean_sd(&st).0, mean_sd(&rad).0, mean_sd(&low).0 * 100.0, wins, seeds
                );
            }
            println!("\n(failure = held-out error > 45 deg; steps->30deg counts non-reaching runs as 1.25 x steps; wins = seeds where error is lower than the v0.1 baseline on the same seed)");
            Ok(())
        }
        other => Err(format!("unknown command '{other}' (try --help)")),
    }
}

/// The wiring graph a network realised (mode-specific), as a Connectome for topology analysis.
fn wiring_connectome(net: &ConnectomeRnn, conn: &Connectome) -> Connectome {
    let r = net.n_regions;
    let mut w = vec![0.0; r * r];
    for e in &net.edges {
        let (i, j) = (net.region_of(e.post), net.region_of(e.pre));
        if i != j && net.is_excitatory(e.pre) && net.is_excitatory(e.post) {
            w[i * r + j] = 1.0;
        }
    }
    Connectome::new(conn.regions.clone(), w.clone(), vec![1.0; r * r], "wiring").unwrap_or_else(|_| conn.clone())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
