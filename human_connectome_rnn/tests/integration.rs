use hcrnn::network::default_roles;
use hcrnn::rng::Rng;
use hcrnn::train::{evaluate, make_trials, train};
use hcrnn::{Connectome, ConnectomeRnn, MaskMode, MemoryGuidedTask, RnnConfig, TrainConfig};

fn small_conn() -> Connectome {
    Connectome::synthetic_human(8, 0.3, 3).unwrap().log_compressed()
}

fn small_cfg(conn: &Connectome, mode: MaskMode) -> RnnConfig {
    let (i, o) = default_roles(conn).unwrap();
    RnnConfig { mode, input_regions: i, output_regions: o, seed: 5, ..Default::default() }
}

#[test]
fn synthetic_connectome_is_valid() {
    let c = Connectome::synthetic_human(40, 0.25, 1).unwrap();
    let n = c.n();
    assert_eq!(n, 80);
    for i in 0..n {
        assert_eq!(c.weight(i, i), 0.0);
        for j in 0..n {
            assert_eq!(c.weight(i, j), c.weight(j, i));
            assert!(c.weight(i, j) >= 0.0);
        }
    }
    assert!(c.is_connected());
    assert!((c.density() - 0.25).abs() < 0.04, "density {}", c.density());
    // left-right homotopic links dominate interhemispheric strength per edge
    assert!(c.interhemispheric_fraction() > 0.05 && c.interhemispheric_fraction() < 0.5);
    // every edge has a positive fibre length
    for i in 0..n {
        for j in 0..n {
            if i != j {
                assert!(c.length(i, j) > 0.0);
            }
        }
    }
}

#[test]
fn degree_preserving_rewire_keeps_degrees() {
    let c = Connectome::synthetic_human(20, 0.25, 2).unwrap();
    let mut rng = Rng::new(9);
    let r = c.degree_preserving_rewire(&mut rng, 20);
    assert_eq!(c.num_edges(), r.num_edges());
    let mut changed = 0;
    for i in 0..c.n() {
        assert_eq!(c.degree(i), r.degree(i), "degree of region {i}");
        for j in 0..c.n() {
            if (c.weight(i, j) > 0.0) != (r.weight(i, j) > 0.0) {
                changed += 1;
            }
        }
    }
    assert!(changed > c.num_edges(), "graph was barely rewired ({changed})");
    let rr = c.density_matched_random(&mut rng);
    assert_eq!(c.num_edges(), rr.num_edges());
}

#[test]
fn constraints_hold_in_every_mode_and_after_training() {
    let c = small_conn();
    for mode in MaskMode::ALL {
        let mut net = ConnectomeRnn::new(&c, small_cfg(&c, mode)).unwrap();
        net.check_constraints().unwrap();
        let task = MemoryGuidedTask::default();
        let cfg = TrainConfig { steps: 5, batch: 4, log_every: 5, eval_trials: 4, threads: 2, ..Default::default() };
        train(&mut net, &task, &cfg, |_| {});
        net.check_constraints().unwrap();
        // spectral radius at init matches the requested gain
        let fresh = ConnectomeRnn::new(&c, small_cfg(&c, mode)).unwrap();
        assert!((fresh.weight_radius() - 0.9).abs() < 0.05, "{mode:?}: {}", fresh.weight_radius());
    }
}

#[test]
fn connectome_mask_is_respected_exactly() {
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let (r, n) = (net.n_regions, net.n_units);
    let m = net.dense_weights();
    for post in 0..n {
        for pre in 0..n {
            let (rp, rq) = (post % r, pre % r);
            if rp != rq {
                let v = m[post * n + pre];
                if c.weight(rp, rq) == 0.0 {
                    assert_eq!(v, 0.0, "weight where the connectome has no tract");
                } else if pre < r {
                    assert!(v > 0.0, "tract present but no excitatory projection");
                } else {
                    assert_eq!(v, 0.0, "inhibitory long-range projection");
                }
            }
        }
    }
}

#[test]
fn delays_follow_fibre_length() {
    let c = small_conn();
    let cfg = small_cfg(&c, MaskMode::Connectome);
    let net = ConnectomeRnn::new(&c, cfg.clone()).unwrap();
    for e in &net.edges {
        let (rp, rq) = (net.region_of(e.post), net.region_of(e.pre));
        if rp == rq {
            assert_eq!(e.delay, 0);
        } else {
            let expect = ((c.length(rp, rq) / (cfg.velocity_mm_per_ms * cfg.dt_ms)).round() as usize).max(1);
            assert_eq!(e.delay, expect);
        }
    }
    let nd = ConnectomeRnn::new(&c, RnnConfig { use_delays: false, ..cfg }).unwrap();
    assert!(nd.edges.iter().all(|e| e.delay == 0));
    assert!(net.max_delay() >= 2);
}

#[test]
fn analytic_gradient_matches_finite_differences() {
    let c = small_conn();
    let mut net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    // perturb so biases/readout/input weights are non-trivial
    let mut rng = Rng::new(11);
    for p in net.params.iter_mut() {
        *p += 0.05 * rng.normal();
    }
    let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 6, delay_max: 6, resp_steps: 6, grace: 1, cue_amp: 1.0 };
    let trial = task.sample(&mut rng);
    let w = net.effective_weights();
    let mut grad = vec![0.0; net.params.len()];
    let loss0 = net.loss_and_grad_with(&w, &trial, &mut grad);
    assert!(loss0.is_finite() && loss0 > 0.0);

    let loss_at = |net: &ConnectomeRnn| {
        let ro = net.forward(&trial);
        net.trial_loss(&ro, &trial)
    };
    let mut checked = 0;
    let mut worst: f64 = 0.0;
    // check a spread of parameters from every block, always including the largest gradients
    let mut idx: Vec<usize> = (0..net.params.len()).step_by(7).collect();
    let mut by_mag: Vec<usize> = (0..net.params.len()).collect();
    by_mag.sort_by(|&a, &b| grad[b].abs().partial_cmp(&grad[a].abs()).unwrap());
    idx.extend(by_mag.into_iter().take(25));
    for &k in &idx {
        let eps = 1e-6;
        let orig = net.params[k];
        net.params[k] = orig + eps;
        let lp = loss_at(&net);
        net.params[k] = orig - eps;
        let lm = loss_at(&net);
        net.params[k] = orig;
        let fd = (lp - lm) / (2.0 * eps);
        let denom = fd.abs().max(grad[k].abs()).max(1e-7);
        let rel = (fd - grad[k]).abs() / denom;
        if fd.abs().max(grad[k].abs()) > 1e-9 {
            worst = worst.max(rel);
            checked += 1;
            assert!(rel < 1e-4, "param {k}: analytic {} vs numeric {fd} (rel {rel})", grad[k]);
        }
    }
    assert!(checked > 30, "only {checked} parameters had non-negligible gradients");
    eprintln!("gradient check: {checked} params, worst relative error {worst:.2e}");
}

#[test]
fn batch_gradient_is_thread_count_invariant() {
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 5, delay_max: 8, resp_steps: 6, grace: 1, cue_amp: 1.0 };
    let trials = make_trials(&task, 7, 3);
    let (mut g1, mut g4) = (vec![0.0; net.params.len()], vec![0.0; net.params.len()]);
    let l1 = net.batch_loss_and_grad(&trials, &mut g1, 1);
    let l4 = net.batch_loss_and_grad(&trials, &mut g4, 4);
    assert!((l1 - l4).abs() < 1e-12);
    for (a, b) in g1.iter().zip(&g4) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn params_roundtrip() {
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let path = std::env::temp_dir().join("hcrnn_roundtrip_test.bin");
    net.save_params(&path).unwrap();
    let mut other = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    other.params.iter_mut().for_each(|p| *p = 0.0);
    other.load_params(&path).unwrap();
    assert_eq!(net.params, other.params);
    let _ = std::fs::remove_file(path);
}

#[test]
fn training_reduces_loss() {
    let c = Connectome::synthetic_human(24, 0.25, 4).unwrap().log_compressed();
    let mut net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let task = MemoryGuidedTask { fix_steps: 4, cue_steps: 6, delay_min: 8, delay_max: 12, resp_steps: 10, grace: 2, cue_amp: 1.0 };
    let before = evaluate(&net, &make_trials(&task, 32, 99)).loss;
    let cfg = TrainConfig { steps: 300, batch: 16, lr: 5e-3, log_every: 150, eval_trials: 16, threads: 4, ..Default::default() };
    train(&mut net, &task, &cfg, |_| {});
    let after = evaluate(&net, &make_trials(&task, 32, 99)).loss;
    assert!(after < before * 0.6, "loss did not drop enough: {before} -> {after}");
}

#[test]
fn tvb_directory_loads_when_available() {
    let Ok(dir) = std::env::var("HCRNN_TVB66") else { return };
    let c = Connectome::from_tvb_dir(std::path::Path::new(&dir)).unwrap();
    assert_eq!(c.n(), 66);
    assert!(c.is_connected());
    let (i, o) = default_roles(&c).unwrap();
    assert!(!i.is_empty() && !o.is_empty());
    let net = ConnectomeRnn::new(&c.log_compressed(), small_cfg(&c, MaskMode::Connectome)).unwrap();
    net.check_constraints().unwrap();
}

// ---------------------------------------------------------------------------
// v0.2: Krylov stability control, topology, harmonics, energy regulariser
// ---------------------------------------------------------------------------

use hcrnn::network::Calibration;
use hcrnn::spectral::{calibrate_scale, delayed_spectral_radius, DelayedRecurrence};
use hcrnn::topology::{harmonics, sym_laplacian, topology_report, connectome_hypergraph, spectral_modules};
use hcrnn::train::{harmonic_profile, lr_at};
use krylov_ds::LinearOperator;

fn arnoldi_cfg(conn: &Connectome, target: f64) -> RnnConfig {
    RnnConfig { calibration: Calibration::Arnoldi, target_radius: target, ..small_cfg(conn, MaskMode::Connectome) }
}

fn dense_radius(net: &ConnectomeRnn) -> f64 {
    let w = net.effective_weights();
    let op = DelayedRecurrence::new(&net.edges, &w, net.n_units, net.alpha);
    let d = op.dim();
    let mut m = nalgebra::DMatrix::<f64>::zeros(d, d);
    let (mut x, mut y) = (vec![0.0; d], vec![0.0; d]);
    for j in 0..d {
        x.iter_mut().for_each(|v| *v = 0.0);
        x[j] = 1.0;
        op.apply(&x, &mut y);
        for i in 0..d {
            m[(i, j)] = y[i];
        }
    }
    m.complex_eigenvalues().iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0.0, f64::max)
}

#[test]
fn arnoldi_radius_matches_dense_eigendecomposition() {
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, arnoldi_cfg(&c, 0.99)).unwrap();
    assert!(net.max_delay() >= 2, "test needs real delays");
    let (dense, kry) = (dense_radius(&net), net.spectral_radius());
    assert!((dense - kry).abs() < 1e-6 * dense.max(1.0), "dense {dense} vs arnoldi {kry}");
    assert!((dense - 0.99).abs() < 1e-3, "calibration missed target: {dense}");
}

#[test]
fn arnoldi_calibration_hits_target_in_every_mode() {
    let c = small_conn();
    for mode in MaskMode::ALL {
        for target in [0.9, 1.0, 1.1] {
            let net = ConnectomeRnn::new(&c, RnnConfig { mode, ..arnoldi_cfg(&c, target) }).unwrap();
            let r = dense_radius(&net);
            assert!((r - target).abs() < 2e-3, "{mode:?} target {target}: dense radius {r}");
            net.check_constraints().unwrap();
        }
    }
}

#[test]
fn calibration_rejects_target_below_leak() {
    let c = small_conn();
    let cfg = arnoldi_cfg(&c, 0.5); // leak factor is 1 - dt/tau = 0.75
    assert!(ConnectomeRnn::new(&c, cfg).is_err());
    let net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let w = net.effective_weights();
    assert!(calibrate_scale(&net.edges, &w, net.n_units, net.alpha, 0.5, 40, 1).is_err());
}

#[test]
fn radius_cap_projects_down_and_leaves_stable_nets_alone() {
    let c = small_conn();
    let mut net = ConnectomeRnn::new(&c, arnoldi_cfg(&c, 1.3)).unwrap();
    let before = net.enforce_radius_cap(1.05);
    assert!((before - 1.3).abs() < 2e-3);
    assert!((net.spectral_radius() - 1.05).abs() < 2e-3);
    net.check_constraints().unwrap();
    let p = net.params.clone();
    net.enforce_radius_cap(1.05 + 0.1);
    assert_eq!(p, net.params, "a net under the cap must not be modified");
}

#[test]
fn arnoldi_converged_on_real_connectome() {
    let Ok(dir) = std::env::var("HCRNN_TVB66") else { return };
    let c = Connectome::from_tvb_dir(std::path::Path::new(&dir)).unwrap().log_compressed();
    let net = ConnectomeRnn::new(&c, RnnConfig { calibration: Calibration::Arnoldi, ..small_cfg(&c, MaskMode::Connectome) }).unwrap();
    let w = net.effective_weights();
    let r = |m: usize| delayed_spectral_radius(&net.edges, &w, net.n_units, net.alpha, m, 7);
    let (r80, r400) = (r(80), r(400));
    assert!((r80 - r400).abs() < 1e-3, "Krylov dim 80 -> {r80}, 400 -> {r400}");
    eprintln!("real connectome: radius m=80 {r80:.6}  m=400 {r400:.6}");
}

#[test]
fn energy_gradient_matches_finite_differences() {
    let c = small_conn();
    let mut net = ConnectomeRnn::new(&c, arnoldi_cfg(&c, 1.0)).unwrap();
    let mut rng = Rng::new(21);
    for p in net.params.iter_mut() {
        *p += 0.05 * rng.normal();
    }
    net.energy_lambda = 0.7;
    let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 6, delay_max: 6, resp_steps: 6, grace: 1, cue_amp: 1.0 };
    let trial = task.sample(&mut rng);
    let w = net.effective_weights();
    let mut grad = vec![0.0; net.params.len()];
    let loss0 = net.loss_and_grad_with(&w, &trial, &mut grad);
    let ro = net.forward(&trial);
    assert!((loss0 - net.objective(&ro, &trial)).abs() < 1e-12);
    assert!(net.mean_energy(&ro) > 0.0);
    let f = |net: &ConnectomeRnn| net.objective(&net.forward(&trial), &trial);
    let mut idx: Vec<usize> = (0..net.params.len()).step_by(5).collect();
    let mut by_mag: Vec<usize> = (0..net.params.len()).collect();
    by_mag.sort_by(|&a, &b| grad[b].abs().partial_cmp(&grad[a].abs()).unwrap());
    idx.extend(by_mag.into_iter().take(25));
    let mut checked = 0;
    for &k in &idx {
        let eps = 1e-6;
        let o = net.params[k];
        net.params[k] = o + eps;
        let lp = f(&net);
        net.params[k] = o - eps;
        let lm = f(&net);
        net.params[k] = o;
        let fd = (lp - lm) / (2.0 * eps);
        if fd.abs().max(grad[k].abs()) > 1e-9 {
            let rel = (fd - grad[k]).abs() / fd.abs().max(grad[k].abs()).max(1e-7);
            assert!(rel < 1e-4, "param {k}: analytic {} vs numeric {fd}", grad[k]);
            checked += 1;
        }
    }
    assert!(checked > 40, "{checked}");
    // the regulariser really changes the gradient
    let mut g0 = vec![0.0; net.params.len()];
    net.energy_lambda = 0.0;
    net.loss_and_grad_with(&w, &trial, &mut g0);
    assert!(g0.iter().zip(&grad).any(|(a, b)| (a - b).abs() > 1e-9));
}

#[test]
fn laplacian_comes_from_spectral_hypergraph_and_harmonics_are_orthonormal() {
    let c = small_conn();
    let n = c.n();
    let hg = connectome_hypergraph(&c).unwrap();
    assert_eq!(hg.num_vertices(), n);
    assert_eq!(hg.num_hyperedges(), c.num_edges());
    // direct L = I - D^-1/2 A D^-1/2 must equal 2 x the system's hypergraph Laplacian
    let l = sym_laplacian(&c);
    for i in 0..n {
        for j in 0..n {
            let expect = if i == j {
                1.0
            } else {
                -c.weight(i, j) / (c.strength(i) * c.strength(j)).sqrt()
            };
            assert!((l[i * n + j] - expect).abs() < 1e-9, "L[{i}][{j}]");
        }
    }
    let h = harmonics(&c);
    assert!(h.values[0].abs() < 1e-9, "lowest harmonic must be the constant mode");
    assert!(h.values.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    assert!(*h.values.last().unwrap() <= 2.0 + 1e-9);
    let gram = h.vectors.transpose() * &h.vectors;
    for i in 0..n {
        for j in 0..n {
            let e = if i == j { 1.0 } else { 0.0 };
            assert!((gram[(i, j)] - e).abs() < 1e-8);
        }
    }
}

#[test]
fn nbsc_topology_matches_closed_forms() {
    // complete graph K_n (degree d = n-1): Hashimoto radius d-1 = n-2, adjacency norm d = n-1
    let c = small_conn().dense_uniform();
    let t = topology_report(&c);
    let d = (c.n() - 1) as f64;
    assert!((t.rho_b - (d - 1.0)).abs() < 1e-6 * d, "rho_B {} vs {}", t.rho_b, d - 1.0);
    assert!((t.adj_norm - d).abs() < 1e-6 * d, "||A|| {} vs {d}", t.adj_norm);
    assert!((t.expansion_ratio - d / (d - 1.0)).abs() < 1e-6);
    // a sparse human-like graph has a non-backtracking radius well above 1
    let t2 = topology_report(&Connectome::synthetic_human(20, 0.25, 2).unwrap());
    assert!(t2.rho_b > 1.0 && t2.adj_norm.is_finite(), "{t2:?}");
}

#[test]
fn spectral_modules_split_two_disconnected_hemispheres() {
    // block-diagonal connectome: two cliques -> 2 spectral modules must match the blocks
    let base = Connectome::synthetic_human(8, 0.3, 5).unwrap();
    let n = base.n();
    let h = n / 2;
    let mut w = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j && (i < h) == (j < h) {
                w[i * n + j] = base.weight(i, j).max(0.5);
            }
        }
    }
    let c = Connectome::new(base.regions.clone(), w, vec![10.0; n * n], "two blocks").unwrap();
    let a = spectral_modules(&c, 2, 1).unwrap();
    assert!((0..h).all(|i| a[i] == a[0]) && (h..n).all(|i| a[i] == a[h]) && a[0] != a[h], "{a:?}");
}

#[test]
fn harmonic_profile_fractions_are_consistent() {
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, arnoldi_cfg(&c, 1.0)).unwrap();
    let task = MemoryGuidedTask::default();
    let trials = make_trials(&task, 24, 2);
    let hp = harmonic_profile(&net, &trials, &harmonics(&c));
    assert!((hp.low + hp.mid + hp.high - 1.0).abs() < 1e-9);
    assert!(hp.low >= 0.0 && hp.high >= 0.0 && hp.mid >= -1e-12);
    assert!(hp.dirichlet_quotient >= -1e-9 && hp.dirichlet_quotient <= 2.0 + 1e-9);
}

#[test]
fn lr_schedule_shape() {
    let cfg = TrainConfig { steps: 100, lr: 1.0, warmup: 10, cosine: true, ..Default::default() };
    assert!((lr_at(&cfg, 5) - 0.5).abs() < 1e-12);
    assert!((lr_at(&cfg, 10) - 1.0).abs() < 1e-12);
    assert!(lr_at(&cfg, 55) < lr_at(&cfg, 20));
    assert!((lr_at(&cfg, 100) - 0.05).abs() < 1e-9);
    let flat = TrainConfig { steps: 100, lr: 0.3, ..Default::default() };
    assert!((lr_at(&flat, 1) - 0.3).abs() < 1e-12 && (lr_at(&flat, 100) - 0.3).abs() < 1e-12);
}

#[test]
fn projection_and_energy_training_keeps_constraints() {
    let c = small_conn();
    let mut net = ConnectomeRnn::new(&c, arnoldi_cfg(&c, 1.0)).unwrap();
    let cfg = TrainConfig { steps: 20, batch: 4, log_every: 10, eval_trials: 4, threads: 2, energy_lambda: 0.2, radius_cap: Some(1.02), project_every: 2, warmup: 3, cosine: true, ..Default::default() };
    hcrnn::train::train(&mut net, &MemoryGuidedTask::default(), &cfg, |_| {});
    net.check_constraints().unwrap();
    assert!(net.spectral_radius() < 1.02 + 0.2, "radius {}", net.spectral_radius());
}

// ---- v0.3: parcel-feature graphs and Glasser labels ----

fn planted_feature_regions(n: usize) -> (Vec<hcrnn::Region>, Vec<Vec<f64>>) {
    let mut rng = Rng::new(11);
    let mut regions = Vec::new();
    let mut rows = Vec::new();
    for i in 0..n {
        let g = i % 2; // two planted groups with opposite feature profiles
        let sign = if g == 0 { 1.0 } else { -1.0 };
        let base = [1.0, 0.5, -0.5, -1.0, 0.8];
        rows.push(base.iter().map(|b| sign * b + 0.1 * rng.normal()).collect());
        regions.push(hcrnn::Region {
            name: format!("r{i}"),
            hemi: if i < n / 2 { 'L' } else { 'R' },
            system: "Unassigned".into(),
            pos: [i as f64, (i * i % 7) as f64, 0.0],
        });
    }
    (regions, rows)
}

#[test]
fn feature_graph_links_similar_parcels_and_rescales_lengths() {
    let (regions, rows) = planted_feature_regions(20);
    let c = Connectome::from_feature_rows(regions, rows, 4, Some(120.0), "test".into()).unwrap();
    let mut cross = 0.0;
    let mut within = 0.0;
    for i in 0..20 {
        for j in 0..20 {
            if i % 2 == j % 2 { within += c.weight(i, j) } else { cross += c.weight(i, j) }
        }
    }
    assert!(within > 0.0 && cross == 0.0, "kNN should only join same-profile parcels (within {within}, cross {cross})");
    let maxlen = c.lengths().iter().cloned().fold(0.0, f64::max);
    assert!((maxlen - 120.0).abs() < 1e-9, "longest length should be rescaled to 120 mm, got {maxlen}");
    // symmetric, zero diagonal, trainable
    for i in 0..20 {
        assert_eq!(c.weight(i, i), 0.0);
        for j in 0..20 {
            assert!((c.weight(i, j) - c.weight(j, i)).abs() < 1e-12);
        }
    }
}

#[test]
fn feature_graph_rejects_bad_input() {
    let (regions, rows) = planted_feature_regions(10);
    assert!(Connectome::from_feature_rows(regions.clone(), rows.clone(), 0, None, "t".into()).is_err());
    assert!(Connectome::from_feature_rows(regions.clone(), rows.clone(), 10, None, "t".into()).is_err());
    let mut flat = regions;
    flat.iter_mut().for_each(|r| r.pos = [0.0; 3]);
    assert!(Connectome::from_feature_rows(flat, rows, 3, None, "t".into()).is_err(), "positions are required");
}

#[test]
fn glasser_names_map_to_systems() {
    use hcrnn::connectome::glasser_system;
    assert_eq!(glasser_system("L_V1_ROI"), "Visual");
    assert_eq!(glasser_system("R_MST_ROI"), "Visual");
    assert_eq!(glasser_system("R_3b_ROI"), "Somatomotor");
    assert_eq!(glasser_system("L_4_ROI"), "Somatomotor");
    assert_eq!(glasser_system("L_A1_ROI"), "Auditory");
    assert_eq!(glasser_system("L_PI_ROI"), "Unassigned");
    assert_eq!(glasser_system("V2"), "Visual");
}

// ---------------------------------------------------------------------------
// LSTM variant
// ---------------------------------------------------------------------------

use hcrnn::{ConnectomeLstm, LstmConfig};

fn small_lstm(conn: &Connectome, mode: MaskMode) -> ConnectomeLstm {
    ConnectomeLstm::new(conn, small_cfg(conn, mode), LstmConfig::default()).unwrap()
}

#[test]
fn lstm_gradient_matches_finite_differences() {
    let c = small_conn();
    let mut net = small_lstm(&c, MaskMode::Connectome);
    let mut rng = Rng::new(21);
    for p in net.params.iter_mut() {
        *p += 0.05 * rng.normal();
    }
    let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 6, delay_max: 6, resp_steps: 6, grace: 1, cue_amp: 1.0 };
    let trial = task.sample(&mut rng);
    let w = net.effective_weights();
    let mut grad = vec![0.0; net.params.len()];
    let loss0 = net.loss_and_grad_with(&w, &trial, &mut grad);
    assert!(loss0.is_finite() && loss0 > 0.0);

    let loss_at = |net: &ConnectomeLstm| hcrnn::model::masked_loss(&net.forward(&trial).y, &trial);
    assert!((loss_at(&net) - loss0).abs() < 1e-12);
    let mut checked = 0;
    let mut worst: f64 = 0.0;
    let mut idx: Vec<usize> = (0..net.params.len()).step_by(5).collect();
    let mut by_mag: Vec<usize> = (0..net.params.len()).collect();
    by_mag.sort_by(|&a, &b| grad[b].abs().partial_cmp(&grad[a].abs()).unwrap());
    idx.extend(by_mag.into_iter().take(40));
    for &k in &idx {
        let eps = 1e-6;
        let orig = net.params[k];
        net.params[k] = orig + eps;
        let lp = loss_at(&net);
        net.params[k] = orig - eps;
        let lm = loss_at(&net);
        net.params[k] = orig;
        let fd = (lp - lm) / (2.0 * eps);
        // central differences of a loss ~0.1 at eps 1e-6 carry ~1e-10 of round-off noise
        let scale = fd.abs().max(grad[k].abs());
        let diff = (fd - grad[k]).abs();
        if scale > 1e-8 {
            worst = worst.max(diff / scale.max(1e-6));
            checked += 1;
            assert!(diff < 1e-4 * scale + 1e-9, "param {k}: analytic {} vs numeric {fd} (diff {diff})", grad[k]);
        }
    }
    assert!(checked > 60, "only {checked} parameters had non-negligible gradients");
    // every gate's recurrent block must actually have been exercised
    for gate in 0..4 {
        let l = &net.layout;
        let nz = (0..net.base.edges.len()).filter(|&e| grad[l.theta + e * 4 + gate].abs() > 1e-9).count();
        assert!(nz > 0, "gate {gate} received no recurrent gradient");
    }
    eprintln!("lstm gradient check: {checked} params, worst relative error {worst:.2e}");
}

#[test]
fn lstm_batch_gradient_is_thread_count_invariant() {
    let c = small_conn();
    let net = small_lstm(&c, MaskMode::Connectome);
    let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 5, delay_max: 8, resp_steps: 6, grace: 1, cue_amp: 1.0 };
    let trials = make_trials(&task, 7, 3);
    let (mut g1, mut g4) = (vec![0.0; net.params.len()], vec![0.0; net.params.len()]);
    let l1 = net.batch_loss_and_grad(&trials, &mut g1, 1);
    let l4 = net.batch_loss_and_grad(&trials, &mut g4, 4);
    assert!((l1 - l4).abs() < 1e-12);
    for (a, b) in g1.iter().zip(&g4) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn lstm_shares_wiring_with_rnn_and_keeps_constraints_in_every_mode() {
    let c = small_conn();
    for mode in MaskMode::ALL {
        let rnn = ConnectomeRnn::new(&c, small_cfg(&c, mode)).unwrap();
        let mut lstm = small_lstm(&c, mode);
        assert_eq!(rnn.edges.len(), lstm.base.edges.len());
        for (a, b) in rnn.edges.iter().zip(&lstm.base.edges) {
            assert_eq!((a.post, a.pre, a.delay, a.kind), (b.post, b.pre, b.delay, b.kind));
        }
        lstm.check_constraints().unwrap();
        assert_eq!(lstm.num_params(), lstm.layout.total);
        // a few training steps must not break Dale's law or the mask in any gate
        let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 5, delay_max: 8, resp_steps: 6, grace: 1, cue_amp: 1.0 };
        let cfg = TrainConfig { steps: 15, batch: 8, lr: 2e-2, log_every: 15, eval_trials: 8, threads: 2, ..Default::default() };
        train(&mut lstm, &task, &cfg, |_| {});
        lstm.check_constraints().unwrap();
    }
}

#[test]
fn lstm_params_roundtrip_and_rejects_rnn_file() {
    let c = small_conn();
    let net = small_lstm(&c, MaskMode::Connectome);
    let path = std::env::temp_dir().join("hcrnn_lstm_roundtrip_test.bin");
    net.save_params(&path).unwrap();
    let mut other = small_lstm(&c, MaskMode::Connectome);
    other.params.iter_mut().for_each(|p| *p = 0.0);
    other.load_params(&path).unwrap();
    assert_eq!(net.params, other.params);
    let rnn = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let rpath = std::env::temp_dir().join("hcrnn_rnn_for_lstm_test.bin");
    rnn.save_params(&rpath).unwrap();
    assert!(other.load_params(&rpath).is_err());
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(rpath);
}

#[test]
fn lstm_training_reduces_loss() {
    let c = Connectome::synthetic_human(24, 0.25, 4).unwrap().log_compressed();
    let mut net = small_lstm(&c, MaskMode::Connectome);
    let task = MemoryGuidedTask { fix_steps: 4, cue_steps: 6, delay_min: 8, delay_max: 12, resp_steps: 10, grace: 2, cue_amp: 1.0 };
    let before = evaluate(&net, &make_trials(&task, 32, 99)).loss;
    let cfg = TrainConfig { steps: 300, batch: 16, lr: 5e-3, log_every: 150, eval_trials: 16, threads: 4, ..Default::default() };
    train(&mut net, &task, &cfg, |_| {});
    let after = evaluate(&net, &make_trials(&task, 32, 99)).loss;
    assert!(after < before * 0.6, "loss did not drop enough: {before} -> {after}");
}

#[test]
fn generic_evaluate_matches_rnn_trial_loss() {
    // the trait-based evaluate must reproduce the original per-trial loss exactly
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 5, delay_max: 8, resp_steps: 6, grace: 1, cue_amp: 1.0 };
    let trials = make_trials(&task, 6, 8);
    let direct: f64 = trials.iter().map(|t| net.trial_loss(&net.forward(t), t)).sum::<f64>() / 6.0;
    assert!((evaluate(&net, &trials).loss - direct).abs() < 1e-12);
}
