//! Memory-guided response task.
//!
//! A cue direction (angle theta) is shown briefly. After a variable delay, a go
//! signal (fixation input turning off) requires the network to report the cue as
//! (cos theta, sin theta). Holding the cue across the delay requires persistent
//! activity distributed across the connectome.
//!
//! Inputs (3): cue_amp*cos(theta), cue_amp*sin(theta), fixation.
//! Outputs (2): cos(theta), sin(theta) after go; 0 before.

use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Trial {
    pub t: usize,
    pub n_in: usize,
    pub n_out: usize,
    /// t x n_in, row-major
    pub inputs: Vec<f64>,
    /// t x n_out, row-major
    pub targets: Vec<f64>,
    /// per-step loss weight (0 = ignored)
    pub mask: Vec<f64>,
    pub angle: f64,
    /// first step after the cue has ended
    pub cue_end: usize,
    /// step at which the go signal arrives
    pub go: usize,
}

#[derive(Clone, Debug)]
pub struct MemoryGuidedTask {
    pub fix_steps: usize,
    pub cue_steps: usize,
    pub delay_min: usize,
    pub delay_max: usize,
    pub resp_steps: usize,
    /// steps after go that are not scored (response onset)
    pub grace: usize,
    pub cue_amp: f64,
}

impl Default for MemoryGuidedTask {
    fn default() -> Self {
        MemoryGuidedTask {
            fix_steps: 8,
            cue_steps: 12,
            delay_min: 20,
            delay_max: 40,
            resp_steps: 24,
            grace: 4,
            cue_amp: 1.0,
        }
    }
}

impl MemoryGuidedTask {
    pub const N_IN: usize = 3;
    pub const N_OUT: usize = 2;

    pub fn sample(&self, rng: &mut Rng) -> Trial {
        let angle = rng.range(0.0, 2.0 * std::f64::consts::PI);
        self.with_angle(angle, rng)
    }

    pub fn with_angle(&self, angle: f64, rng: &mut Rng) -> Trial {
        let delay = if self.delay_max > self.delay_min {
            self.delay_min + rng.below(self.delay_max - self.delay_min + 1)
        } else {
            self.delay_min
        };
        let cue_start = self.fix_steps;
        let cue_end = cue_start + self.cue_steps;
        let go = cue_end + delay;
        let t = go + self.resp_steps;
        let (n_in, n_out) = (Self::N_IN, Self::N_OUT);
        let mut inputs = vec![0.0; t * n_in];
        let mut targets = vec![0.0; t * n_out];
        let mut mask = vec![0.0; t];
        let (c, s) = (angle.cos(), angle.sin());
        for step in 0..t {
            if step >= cue_start && step < cue_end {
                inputs[step * n_in] = self.cue_amp * c;
                inputs[step * n_in + 1] = self.cue_amp * s;
            }
            if step < go {
                inputs[step * n_in + 2] = 1.0;
                mask[step] = 0.5; // keep the readout silent until go
            } else if step >= go + self.grace {
                targets[step * n_out] = c;
                targets[step * n_out + 1] = s;
                mask[step] = 1.0;
            }
        }
        Trial { t, n_in, n_out, inputs, targets, mask, angle, cue_end, go }
    }
}
