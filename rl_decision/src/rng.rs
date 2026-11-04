//! Tiny deterministic PRNG (xorshift64*). Deliberately not `rand`: the
//! workspace already carries two incompatible `rand` majors (see
//! `hg_builder/Cargo.toml`) and this layer needs reproducibility, not
//! cryptographic strength.

#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix the seed so small seeds (0, 1, 2...) give distinct streams
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self(if z == 0 { 0x1234_5678_9ABC_DEF1 } else { z })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Index drawn from an (unnormalised, non-negative) weight vector.
    pub fn categorical(&mut self, w: &[f64]) -> usize {
        let total: f64 = w.iter().map(|x| x.max(0.0)).sum();
        if !(total > 0.0) {
            return (self.next_u64() as usize) % w.len().max(1);
        }
        let mut t = self.uniform() * total;
        for (i, x) in w.iter().enumerate() {
            t -= x.max(0.0);
            if t <= 0.0 {
                return i;
            }
        }
        w.len() - 1
    }
}
