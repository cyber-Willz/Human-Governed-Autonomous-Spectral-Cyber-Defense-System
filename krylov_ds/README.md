# krylov_ds

Production-ready Arnoldi and Lanczos Krylov subspace methods in Rust.

Both methods build an orthonormal basis of a Krylov subspace
`K_k(A, v0) = span{v0, A v0, ..., A^(k-1) v0}` and a small dense projection
of `A` onto that subspace, from which approximate eigenvalues ("Ritz
values") of `A` can be extracted cheaply. This is the standard approach for
large-scale (sparse or matrix-free) eigenvalue problems where forming or
directly diagonalizing `A` is infeasible.

## What's implemented

- **`Arnoldi`** — for general (non-symmetric) operators. Builds an upper
  Hessenberg projection using modified Gram-Schmidt with **DGKS selective
  reorthogonalization** (a second MGS pass triggered only when the
  Daniel-Gragg-Kaufman-Stewart criterion `‖w‖ / ‖w_pre‖ ≤ 0.717` indicates
  a step lost accuracy) — the standard robust choice, avoiding both the
  cost of always-reorthogonalizing and the fragility of never doing so.
- **`Lanczos`** — for symmetric operators. Three-term recurrence producing
  a tridiagonal projection, with a `Reorthogonalization` policy
  (`None` or `Full`). Full reorthogonalization is recommended for
  production use since the short recurrence loses orthogonality quickly
  once any Ritz value converges, producing spurious "ghost" duplicate
  eigenvalues if uncorrected.
- **Happy breakdown detection** — if the residual vector at any step is
  numerically zero, the Krylov subspace is exactly `A`-invariant and every
  Ritz value is an *exact* eigenvalue; both methods detect and report this
  rather than dividing by (near) zero.
- **`LinearOperator` trait** — the methods only need a matrix-vector
  product, so they work with the provided `DenseMatrix` and `CsrMatrix`
  types, or with any matrix-free operator you implement yourself
  (stencils, FFT-based operators, etc).
- **Ritz value/vector extraction** (`eig` module) — solves the small dense
  projected eigenproblem via `nalgebra` (general complex eigenvalues for
  Hessenberg, symmetric eigendecomposition for tridiagonal) and lifts
  eigenvectors back to `R^n`, along with a cheap a-posteriori residual
  bound `‖A x - λx‖` computed from the last Hessenberg/tridiagonal row
  without forming `Ax` explicitly.

## What's intentionally out of scope (v0.1)

- **Implicit restarting** (IRAM/thick-restart Lanczos). For subspaces that
  need to grow beyond available memory, or for isolating specific
  eigenvalue clusters via polynomial filtering, add a restart wrapper on
  top of `Arnoldi`/`Lanczos` — the projection outputs (`h`/`alpha`+`beta`,
  plus `q`) are exactly what a restart scheme needs to construct a filtered
  starting vector.
- **Complex-valued operators.** Currently real (`f32`/`f64` via
  `num_traits::Float`) only. Extending to `Complex<T>` mainly requires
  swapping `dot` for a conjugate inner product and is a natural follow-up
  if you need Hermitian/non-Hermitian complex problems.
- **GMRES / linear-solve usage of Arnoldi.** This crate exposes the
  Arnoldi factorization itself; building GMRES on top only needs a small
  least-squares solve against `H` (via Givens rotations or `nalgebra`),
  not a new factorization.

## Example

```rust
use krylov_ds::{Arnoldi, DenseMatrix};
use krylov_ds::eig::arnoldi_ritz_values;

let a = DenseMatrix::from_fn(4, |i, j| if i == j { (i + 1) as f64 } else { 0.1 });
let v0 = vec![1.0, 0.0, 0.0, 0.0];
let result = Arnoldi::new(4, 1e-12).run(&a, &v0).unwrap();
let ritz = arnoldi_ritz_values(&result);
```

For a symmetric operator (e.g. a graph Laplacian) built from a sparse
`CsrMatrix`:

```rust
use krylov_ds::{CsrMatrix, Lanczos, Reorthogonalization};
use krylov_ds::eig::lanczos_ritz_pairs;

let triplets = vec![(0usize, 0usize, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)];
let a = CsrMatrix::from_triplets(2, &triplets);
let v0 = vec![1.0, 0.5];
let result = Lanczos::new(2, 1e-12, Reorthogonalization::Full).run(&a, &v0).unwrap();
let ritz = lanczos_ritz_pairs(&result);
for pair in &ritz {
    println!("lambda = {:.6}, residual = {:.2e}", pair.value, pair.residual_norm);
}
```

## Testing

```
cargo test
```

The integration test suite (`tests/integration_test.rs`) validates results
against `nalgebra`'s direct dense eigensolvers on both symmetric and
non-symmetric matrices, checks CSR vs. dense operator equivalence, checks
happy-breakdown handling on an exact eigenvector, checks error handling for
dimension mismatches / oversized subspaces, and checks that full
reorthogonalization avoids ghost eigenvalues on a clustered-spectrum
matrix run to full dimension.

MSRV: 1.75 (matches the pin used elsewhere in this toolchain, e.g.
`causal_llm`).
