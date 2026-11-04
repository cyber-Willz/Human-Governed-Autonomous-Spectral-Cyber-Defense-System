use burn::tensor::{backend::Backend, Tensor};

use crate::config::{RopeConfig, RopeScaling};

/// Inverse frequencies (`head_dim / 2` of them) and the post-scaling factor applied to
/// cos/sin, for any supported [`RopeScaling`]. Formulas follow transformers v5.9.0
/// `modeling_rope_utils.py` (`_compute_linear_scaling_rope_parameters`,
/// `_compute_llama3_parameters`, `_compute_yarn_parameters`).
pub fn inv_frequencies(head_dim: usize, rope: &RopeConfig) -> (Vec<f64>, f64) {
    assert!(head_dim % 2 == 0, "RoPE requires an even head_dim");
    let half = head_dim / 2;
    let dim = head_dim as f64;
    let base = rope.theta;

    // pos_freqs_i = base^(2i/dim)
    let pos_freqs: Vec<f64> = (0..half).map(|i| base.powf((2 * i) as f64 / dim)).collect();
    let inv: Vec<f64> = pos_freqs.iter().map(|p| 1.0 / p).collect();

    match &rope.scaling {
        RopeScaling::None => (inv, 1.0),

        RopeScaling::Linear { factor } => (inv.iter().map(|f| f / factor).collect(), 1.0),

        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        } => {
            let old = *original_max_position_embeddings as f64;
            let low_wavelen = old / low_freq_factor;
            let high_wavelen = old / high_freq_factor;
            let out = inv
                .iter()
                .map(|&f| {
                    let wavelen = 2.0 * std::f64::consts::PI / f;
                    // wavelen < high_wavelen: unchanged; > low_wavelen: divided by factor;
                    // in between: smooth interpolation of the two.
                    let scaled = if wavelen > low_wavelen { f / factor } else { f };
                    let is_medium = !(wavelen < high_wavelen) && !(wavelen > low_wavelen);
                    if is_medium {
                        let smooth = (old / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * scaled / factor + smooth * scaled
                    } else {
                        scaled
                    }
                })
                .collect();
            (out, 1.0)
        }

        RopeScaling::Yarn {
            factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
            attention_factor,
            truncate,
        } => {
            let orig = *original_max_position_embeddings as f64;
            let find_correction_dim = |num_rotations: f64| {
                dim * (orig / (num_rotations * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * base.ln())
            };
            let mut low = find_correction_dim(*beta_fast);
            let mut high = find_correction_dim(*beta_slow);
            if *truncate {
                low = low.floor();
                high = high.ceil();
            }
            let low = low.max(0.0);
            let high = high.min(dim - 1.0);
            // HF: `if min == max: max += 0.001` (avoid a singular ramp).
            let high = if low == high { high + 0.001 } else { high };

            let out = (0..half)
                .map(|i| {
                    let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                    let extrapolation_factor = 1.0 - ramp;
                    let interpolation = 1.0 / (factor * pos_freqs[i]);
                    let extrapolation = inv[i];
                    interpolation * (1.0 - extrapolation_factor)
                        + extrapolation * extrapolation_factor
                })
                .collect();

            let af = attention_factor.unwrap_or(if *factor <= 1.0 {
                1.0
            } else {
                0.1 * factor.ln() + 1.0
            });
            (out, af)
        }
    }
}

/// Precomputed rotary cos/sin tables, shape `[capacity, head_dim]`.
///
/// Built on the CPU in f64 and rounded once to f32 (transformers computes the angle product
/// in f32, so long-context tables differ from HF's at the ~1e-3 rad level; the f64 path is
/// the more accurate of the two). The table is sized to the *requested* context, not to
/// `max_position_embeddings`: a 131072-position Llama-3.1 table at head_dim 128 would
/// otherwise cost ~134 MB even for a 200-token chat.
pub struct RotaryCache<B: Backend> {
    pub cos: Tensor<B, 2>,
    pub sin: Tensor<B, 2>,
    capacity: usize,
}

impl<B: Backend> RotaryCache<B> {
    pub fn new(head_dim: usize, capacity: usize, rope: &RopeConfig, device: &B::Device) -> Self {
        assert!(capacity > 0, "RotaryCache capacity must be positive");
        let half = head_dim / 2;
        let (inv, attention_factor) = inv_frequencies(head_dim, rope);

        let mut cos = vec![0f32; capacity * head_dim];
        let mut sin = vec![0f32; capacity * head_dim];
        for p in 0..capacity {
            for (i, f) in inv.iter().enumerate() {
                let (s, c) = (p as f64 * f).sin_cos();
                let (c, s) = ((c * attention_factor) as f32, (s * attention_factor) as f32);
                // HF layout: cat([freqs, freqs], -1)
                cos[p * head_dim + i] = c;
                cos[p * head_dim + i + half] = c;
                sin[p * head_dim + i] = s;
                sin[p * head_dim + i + half] = s;
            }
        }
        Self {
            cos: Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([capacity, head_dim]),
            sin: Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([capacity, head_dim]),
            capacity,
        }
    }

    /// Number of positions this table covers.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Rows `[start, start+seq_len)`, e.g. offset by the KV-cache length for a decode step.
    pub fn slice(&self, start: usize, seq_len: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
        assert!(
            start + seq_len <= self.capacity,
            "position {}..{} exceeds the RoPE table ({} positions); pass a larger max_seq to \
             init_generation",
            start,
            start + seq_len,
            self.capacity
        );
        let dim = self.cos.dims()[1];
        (
            self.cos.clone().slice([start..start + seq_len, 0..dim]),
            self.sin.clone().slice([start..start + seq_len, 0..dim]),
        )
    }
}

/// `cat(-x[..., d/2:], x[..., :d/2])`, HF's `rotate_half`.
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, t, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..t, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..t, half..d]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

/// Applies RoPE to a `[B, H, T, D]` tensor given `[T, D]` cos/sin (broadcast over B and H).
pub fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [_, _, t, d] = x.dims();
    let cos = cos.clone().reshape([1, 1, t, d]);
    let sin = sin.clone().reshape([1, 1, t, d]);
    x.clone() * cos + rotate_half(x) * sin
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type B = NdArray<f32>;

    fn plain(theta: f64) -> RopeConfig {
        RopeConfig { theta, scaling: RopeScaling::None }
    }
    fn vec_of(t: Tensor<B, 2>) -> Vec<f32> {
        t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
    }

    #[test]
    fn default_inv_freq_formula() {
        let (inv, af) = inv_frequencies(8, &plain(10000.0));
        for (i, f) in inv.iter().enumerate() {
            assert!((f - 10000f64.powf(-(2.0 * i as f64) / 8.0)).abs() < 1e-15);
        }
        assert_eq!(af, 1.0);
    }

    #[test]
    fn linear_divides_everything_by_factor() {
        let (base, _) = inv_frequencies(16, &plain(10000.0));
        let (lin, _) = inv_frequencies(
            16,
            &RopeConfig { theta: 10000.0, scaling: RopeScaling::Linear { factor: 4.0 } },
        );
        for (a, b) in base.iter().zip(&lin) {
            assert!((a / 4.0 - b).abs() < 1e-18);
        }
    }

    /// Independent, branch-structured re-derivation of the Llama-3.1 rule (the crate uses the
    /// HF `where`-style formulation) checked on Llama-3.1-8B's real parameters.
    #[test]
    fn llama3_three_bands_match_piecewise_definition() {
        let (theta, hd, factor, lo, hi, old) = (500000.0f64, 128usize, 8.0f64, 1.0f64, 4.0f64, 8192.0f64);
        let (base, _) = inv_frequencies(hd, &plain(theta));
        let (got, af) = inv_frequencies(
            hd,
            &RopeConfig {
                theta,
                scaling: RopeScaling::Llama3 {
                    factor,
                    low_freq_factor: lo,
                    high_freq_factor: hi,
                    original_max_position_embeddings: 8192,
                },
            },
        );
        assert_eq!(af, 1.0);
        let (mut n_high, mut n_mid, mut n_low) = (0, 0, 0);
        for (f, g) in base.iter().zip(&got) {
            let wl = 2.0 * std::f64::consts::PI / f;
            let want = if wl < old / hi {
                n_high += 1;
                *f
            } else if wl > old / lo {
                n_low += 1;
                f / factor
            } else {
                n_mid += 1;
                let s = (old / wl - lo) / (hi - lo);
                (1.0 - s) * f / factor + s * f
            };
            assert!((want - g).abs() <= 1e-15 * want.abs().max(1.0), "{want} vs {g}");
        }
        // All three regimes must actually be exercised by this configuration.
        assert!(n_high > 0 && n_mid > 0 && n_low > 0, "{n_high}/{n_mid}/{n_low}");
        // Frequencies stay monotonically decreasing (a broken blend would not).
        assert!(got.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn yarn_ramp_and_attention_factor() {
        // head_dim 16, theta 1e4, original_max 64, factor 4:
        //   low  = floor(16*ln(64/(32*2π)) / (2 ln 1e4)) = floor(-0.995) = -1 -> clamped to 0
        //   high = ceil (16*ln(64/(1 *2π)) / (2 ln 1e4)) = ceil(2.016)   =  3
        // so dim 0 keeps its frequency, dims >= 3 are fully interpolated (/4), dims 1,2 blend.
        let cfg = RopeConfig {
            theta: 10000.0,
            scaling: RopeScaling::Yarn {
                factor: 4.0,
                original_max_position_embeddings: 64,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: None,
                truncate: true,
            },
        };
        let (base, _) = inv_frequencies(16, &plain(10000.0));
        let (y, af) = inv_frequencies(16, &cfg);
        assert!((y[0] - base[0]).abs() < 1e-15, "highest-frequency dim is untouched");
        for i in 3..8 {
            assert!((y[i] - base[i] / 4.0).abs() < 1e-15, "dim {i} must be interpolated");
        }
        let blend1 = base[1] / 4.0 * (1.0 / 3.0) + base[1] * (2.0 / 3.0);
        assert!((y[1] - blend1).abs() < 1e-15);
        assert!((af - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-15);

        // explicit attention_factor overrides the default
        let mut cfg2 = cfg.clone();
        if let RopeScaling::Yarn { attention_factor, .. } = &mut cfg2.scaling {
            *attention_factor = Some(0.5);
        }
        assert_eq!(inv_frequencies(16, &cfg2).1, 0.5);
    }

    #[test]
    fn cache_position_zero_is_identity_times_attention_factor() {
        let device = NdArrayDevice::default();
        let cfg = RopeConfig {
            theta: 10000.0,
            scaling: RopeScaling::Yarn {
                factor: 4.0,
                original_max_position_embeddings: 64,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: None,
                truncate: true,
            },
        };
        let cache = RotaryCache::<B>::new(16, 8, &cfg, &device);
        let (cos0, sin0) = cache.slice(0, 1);
        let af = (0.1 * 4f64.ln() + 1.0) as f32;
        assert!(vec_of(cos0).iter().all(|c| (c - af).abs() < 1e-6));
        assert!(vec_of(sin0).iter().all(|s| s.abs() < 1e-6));
    }

    #[test]
    fn plain_rope_preserves_norm() {
        let device = NdArrayDevice::default();
        let hd = 8;
        let cache = RotaryCache::<B>::new(hd, 16, &plain(10000.0), &device);
        let (cos, sin) = cache.slice(5, 3);
        let x = Tensor::<B, 4>::random(
            [2, 3, 3, hd],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let before: f32 = x.clone().powf_scalar(2.0).sum().into_scalar();
        let after: f32 = apply_rope(x, &cos, &sin).powf_scalar(2.0).sum().into_scalar();
        assert!((before - after).abs() < 1e-3 * before.max(1.0), "{before} vs {after}");
    }

    #[test]
    #[should_panic(expected = "exceeds the RoPE table")]
    fn slice_past_capacity_is_a_clear_error() {
        let device = NdArrayDevice::default();
        let cache = RotaryCache::<B>::new(4, 4, &plain(10000.0), &device);
        let _ = cache.slice(3, 2);
    }
}
