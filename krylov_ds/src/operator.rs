use num_traits::Float;

/// Anything that can act as a matrix in a Krylov method only needs to supply
/// a matrix-vector product. This makes the methods usable with dense
/// matrices, sparse matrices, or fully matrix-free operators (e.g. a stencil,
/// or an FFT-based convolution).
pub trait LinearOperator<T> {
    /// Dimension `n` of the (square) operator, acting on R^n.
    fn dim(&self) -> usize;

    /// Compute `y = A * x`. Implementations may assume `x.len() == y.len() == self.dim()`.
    fn apply(&self, x: &[T], y: &mut [T]);
}

/// Dense row-major matrix, mainly useful for testing and small problems.
#[derive(Debug, Clone)]
pub struct DenseMatrix<T> {
    pub n: usize,
    pub data: Vec<T>,
}

impl<T: Clone> DenseMatrix<T> {
    pub fn new(n: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), n * n, "data length must equal n*n");
        Self { n, data }
    }

    pub fn from_fn(n: usize, f: impl Fn(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                data.push(f(i, j));
            }
        }
        Self { n, data }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.n + j]
    }
}

impl<T: Float> LinearOperator<T> for DenseMatrix<T> {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[T], y: &mut [T]) {
        let n = self.n;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        for i in 0..n {
            let row = &self.data[i * n..(i + 1) * n];
            let mut acc = T::zero();
            for j in 0..n {
                acc = acc + row[j] * x[j];
            }
            y[i] = acc;
        }
    }
}

/// Compressed Sparse Row matrix, for the common case of sparse operators
/// (e.g. graph Laplacians, discretized PDEs, adjacency-derived matrices).
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<T>,
}

impl<T> CsrMatrix<T> {
    pub fn new(n: usize, row_ptr: Vec<usize>, col_idx: Vec<usize>, values: Vec<T>) -> Self {
        assert_eq!(row_ptr.len(), n + 1, "row_ptr must have length n+1");
        assert_eq!(col_idx.len(), values.len(), "col_idx/values length mismatch");
        Self { n, row_ptr, col_idx, values }
    }

    /// Build from a list of (row, col, value) triplets. Duplicate entries are summed.
    pub fn from_triplets(n: usize, triplets: &[(usize, usize, T)]) -> Self
    where
        T: Float,
    {
        let mut rows: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        for &(r, c, v) in triplets {
            rows[r].push((c, v));
        }
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for row in rows.iter_mut() {
            row.sort_by_key(|&(c, _)| c);
            let mut last_col: Option<usize> = None;
            for (c, v) in row.drain(..) {
                if last_col == Some(c) {
                    let idx = values.len() - 1;
                    values[idx] = values[idx] + v;
                } else {
                    col_idx.push(c);
                    values.push(v);
                    last_col = Some(c);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Self { n, row_ptr, col_idx, values }
    }
}

impl<T: Float> LinearOperator<T> for CsrMatrix<T> {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[T], y: &mut [T]) {
        for i in 0..self.n {
            let mut acc = T::zero();
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                acc = acc + self.values[k] * x[self.col_idx[k]];
            }
            y[i] = acc;
        }
    }
}

#[inline]
pub(crate) fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

#[inline]
pub(crate) fn norm<T: Float>(a: &[T]) -> T {
    dot(a, a).sqrt()
}

#[inline]
pub(crate) fn axpy<T: Float>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = *yi + alpha * xi;
    }
}
