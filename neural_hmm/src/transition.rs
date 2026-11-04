use crate::error::{HmmError, HmmResult};
use serde::{Deserialize, Serialize};

/// Row-stochastic Markov transition matrix: `rows[i][j] = P(state j at t+1 | state i at t)`.
///
/// Constructed only through [`TransitionMatrix::new`] / [`TransitionMatrix::from_json_file`],
/// which validate squareness, non-negativity, and row-stochasticity up front. This means every
/// other piece of code downstream (the filter's predict step) can assume the matrix is valid
/// and skip re-checking it on every tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionMatrix {
    rows: Vec<Vec<f32>>,
    #[serde(default)]
    labels: Vec<String>,
}

const ROW_SUM_TOLERANCE: f32 = 1e-3;

impl TransitionMatrix {
    pub fn new(rows: Vec<Vec<f32>>) -> HmmResult<Self> {
        Self::with_labels(rows, Vec::new())
    }

    pub fn with_labels(rows: Vec<Vec<f32>>, labels: Vec<String>) -> HmmResult<Self> {
        let matrix = Self { rows, labels };
        matrix.validate()?;
        Ok(matrix)
    }

    fn validate(&self) -> HmmResult<()> {
        if self.rows.is_empty() {
            return Err(HmmError::EmptyTransitionMatrix);
        }
        let n = self.rows.len();
        for (i, row) in self.rows.iter().enumerate() {
            if row.len() != n {
                return Err(HmmError::NonSquareTransitionMatrix {
                    row: i,
                    expected: n,
                    actual: row.len(),
                });
            }
            let mut sum = 0.0f32;
            for (j, &p) in row.iter().enumerate() {
                if !p.is_finite() {
                    return Err(HmmError::NonFiniteValue { state: j, value: p });
                }
                if p < 0.0 {
                    return Err(HmmError::NegativeProbability { row: i, col: j, value: p });
                }
                sum += p;
            }
            if (sum - 1.0).abs() > ROW_SUM_TOLERANCE {
                return Err(HmmError::InvalidRowSum { row: i, sum, tolerance: ROW_SUM_TOLERANCE });
            }
        }
        if !self.labels.is_empty() && self.labels.len() != n {
            return Err(HmmError::StateDimensionMismatch { expected: n, actual: self.labels.len() });
        }
        Ok(())
    }

    #[inline]
    pub fn num_states(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    pub fn get(&self, from: usize, to: usize) -> f32 {
        self.rows[from][to]
    }

    pub fn row(&self, from: usize) -> &[f32] {
        &self.rows[from]
    }

    pub fn label(&self, state: usize) -> Option<&str> {
        self.labels.get(state).map(|s| s.as_str())
    }

    pub fn from_json_file(path: &str) -> HmmResult<Self> {
        let data = std::fs::read_to_string(path).map_err(|e| HmmError::ConfigIo(e.to_string()))?;
        let matrix: TransitionMatrix =
            serde_json::from_str(&data).map_err(|e| HmmError::ConfigParse(e.to_string()))?;
        matrix.validate()?;
        Ok(matrix)
    }

    pub fn to_json_file(&self, path: &str) -> HmmResult<()> {
        let data = serde_json::to_string_pretty(self).map_err(|e| HmmError::ConfigParse(e.to_string()))?;
        std::fs::write(path, data).map_err(|e| HmmError::ConfigIo(e.to_string()))
    }

    /// Applies the Markov predict step to a belief vector: `predicted[j] = sum_i prior[i] * P(i -> j)`.
    pub fn predict(&self, prior: &[f32]) -> HmmResult<Vec<f32>> {
        let n = self.num_states();
        if prior.len() != n {
            return Err(HmmError::StateDimensionMismatch { expected: n, actual: prior.len() });
        }
        let mut predicted = vec![0.0f32; n];
        for (next_state, slot) in predicted.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (current_state, &p) in prior.iter().enumerate() {
                sum += p * self.rows[current_state][next_state];
            }
            *slot = sum;
        }
        Ok(predicted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_square() {
        let err = TransitionMatrix::new(vec![vec![1.0, 0.0], vec![0.5, 0.3, 0.2]]).unwrap_err();
        assert!(matches!(err, HmmError::NonSquareTransitionMatrix { .. }));
    }

    #[test]
    fn rejects_bad_row_sum() {
        let err = TransitionMatrix::new(vec![vec![0.5, 0.4], vec![0.5, 0.5]]).unwrap_err();
        assert!(matches!(err, HmmError::InvalidRowSum { .. }));
    }

    #[test]
    fn rejects_negative() {
        let err = TransitionMatrix::new(vec![vec![1.2, -0.2], vec![0.5, 0.5]]).unwrap_err();
        assert!(matches!(err, HmmError::NegativeProbability { .. }));
    }

    #[test]
    fn accepts_valid_and_predicts() {
        let m = TransitionMatrix::new(vec![vec![0.9, 0.1], vec![0.2, 0.8]]).unwrap();
        let predicted = m.predict(&[1.0, 0.0]).unwrap();
        assert!((predicted[0] - 0.9).abs() < 1e-6);
        assert!((predicted[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn round_trips_through_json() {
        let m = TransitionMatrix::with_labels(
            vec![vec![0.9, 0.1], vec![0.2, 0.8]],
            vec!["normal".into(), "attack".into()],
        )
        .unwrap();
        let path = std::env::temp_dir().join("transition_test.json");
        m.to_json_file(path.to_str().unwrap()).unwrap();
        let loaded = TransitionMatrix::from_json_file(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.num_states(), 2);
        assert_eq!(loaded.label(1), Some("attack"));
        let _ = std::fs::remove_file(path);
    }
}
