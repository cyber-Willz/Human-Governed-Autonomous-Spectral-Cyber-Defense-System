use approx::assert_relative_eq;
use nalgebra::DMatrix;

use krylov_ds::eig::{arnoldi_ritz_values, lanczos_ritz_pairs};
use krylov_ds::{Arnoldi, CsrMatrix, DenseMatrix, Lanczos, LinearOperator, Reorthogonalization};

fn sorted_reals(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn arnoldi_full_dimension_matches_direct_eigenvalues_diagonal() {
    // A diagonal matrix's eigenvalues are trivially known; Arnoldi run to
    // full dimension must recover them exactly (up to floating point).
    let n = 6;
    let a = DenseMatrix::from_fn(n, |i, j| if i == j { (i as f64 + 1.0) * 2.0 } else { 0.0 });
    let v0: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.3).collect();

    let result = Arnoldi::new(n, 1e-13).run(&a, &v0).unwrap();
    let ritz = arnoldi_ritz_values(&result);
    let mut got: Vec<f64> = ritz.iter().map(|c| c.re).collect();
    got = sorted_reals(got);
    let expected = sorted_reals((0..n).map(|i| (i as f64 + 1.0) * 2.0).collect());

    for (g, e) in got.iter().zip(expected.iter()) {
        assert_relative_eq!(g, e, epsilon = 1e-8);
    }
}

#[test]
fn arnoldi_matches_dense_eigenvalues_on_random_nonsymmetric_matrix() {
    let n = 8;
    // Deterministic pseudo-random-looking but fixed matrix (reproducible test).
    let a = DenseMatrix::from_fn(n, |i, j| {
        let x = (i as f64 * 7.0 + j as f64 * 13.0).sin();
        if i == j {
            x * 0.3 + (i as f64 + 1.0)
        } else {
            x * 0.5
        }
    });
    let v0: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.7).cos() + 1.0).collect();

    let result = Arnoldi::new(n, 1e-13).run(&a, &v0).unwrap();
    assert_eq!(result.steps, n);
    let ritz = arnoldi_ritz_values(&result);

    let dense = DMatrix::from_fn(n, n, |i, j| *a.get(i, j));
    let direct = dense.complex_eigenvalues();

    let mut got_re = sorted_reals(ritz.iter().map(|c| c.re).collect());
    let mut expected_re = sorted_reals(direct.iter().map(|c| c.re).collect());
    got_re.sort_by(|a, b| a.partial_cmp(b).unwrap());
    expected_re.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (g, e) in got_re.iter().zip(expected_re.iter()) {
        assert_relative_eq!(g, e, epsilon = 1e-6);
    }
}

#[test]
fn lanczos_matches_dense_symmetric_eigenvalues() {
    let n = 10;
    // Symmetric matrix by construction.
    let a = DenseMatrix::from_fn(n, |i, j| {
        if i == j {
            2.0 + i as f64 * 0.1
        } else {
            let v = ((i + j) as f64 * 0.37).sin() * 0.4;
            v
        }
    });
    // Symmetrize explicitly to remove any floating point asymmetry.
    let mut data = a.data.clone();
    for i in 0..n {
        for j in 0..n {
            let avg = 0.5 * (data[i * n + j] + data[j * n + i]);
            data[i * n + j] = avg;
        }
    }
    let a = DenseMatrix::new(n, data);

    let v0: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.13).collect();
    let result = Lanczos::new(n, 1e-13, Reorthogonalization::Full).run(&a, &v0).unwrap();
    assert_eq!(result.steps, n);

    let pairs = lanczos_ritz_pairs(&result);
    let got: Vec<f64> = pairs.iter().map(|p| p.value).collect();

    let dense = DMatrix::from_fn(n, n, |i, j| *a.get(i, j));
    let eig = dense.symmetric_eigenvalues();
    let mut expected: Vec<f64> = eig.iter().copied().collect();
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (g, e) in got.iter().zip(expected.iter()) {
        assert_relative_eq!(g, e, epsilon = 1e-8);
    }

    // Residuals should be tiny at full dimension (subspace is A-invariant).
    for p in &pairs {
        assert!(p.residual_norm < 1e-6, "residual too large: {}", p.residual_norm);
    }
}

#[test]
fn lanczos_partial_run_gives_extremal_ritz_convergence() {
    // Classic use case: k << n, only extremal eigenvalues are needed.
    let n = 60;
    let a = DenseMatrix::from_fn(n, |i, j| {
        if i == j {
            (i as f64 + 1.0).powi(2) // widely spread spectrum: extremal values isolate fast
        } else if (i as i64 - j as i64).abs() == 1 {
            0.5
        } else {
            0.0
        }
    });
    let v0: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.7).sin() + 1.5).collect();

    let result = Lanczos::new(20, 1e-13, Reorthogonalization::Full).run(&a, &v0).unwrap();
    let pairs = lanczos_ritz_pairs(&result);

    let dense = DMatrix::from_fn(n, n, |i, j| *a.get(i, j));
    let eig = dense.symmetric_eigenvalues();
    let mut expected: Vec<f64> = eig.iter().copied().collect();
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // The largest Ritz value should closely approximate the true largest
    // eigenvalue even though we only took 20 of 60 possible steps.
    let largest_ritz = pairs.iter().map(|p| p.value).fold(f64::MIN, f64::max);
    let largest_true = *expected.last().unwrap();
    assert_relative_eq!(largest_ritz, largest_true, max_relative = 1e-4);
}

#[test]
fn full_reorthogonalization_prevents_ghost_eigenvalues() {
    // Without reorthogonalization, Lanczos on a matrix with clustered
    // eigenvalues run for many steps tends to produce duplicate ("ghost")
    // Ritz values once round-off destroys orthogonality. With full
    // reorthogonalization this should not happen: distinct true eigenvalues
    // should not appear as near-duplicates among the Ritz values.
    let n = 30;
    let a = DenseMatrix::from_fn(n, |i, j| {
        if i == j {
            1.0 + (i as f64) * 0.05
        } else if (i as i64 - j as i64).abs() == 1 {
            0.3
        } else {
            0.0
        }
    });
    let v0: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.01).collect();

    let result = Lanczos::new(n, 1e-13, Reorthogonalization::Full).run(&a, &v0).unwrap();
    let pairs = lanczos_ritz_pairs(&result);

    let mut vals: Vec<f64> = pairs.iter().map(|p| p.value).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for w in vals.windows(2) {
        let gap = w[1] - w[0];
        assert!(gap > 1e-9, "unexpectedly close Ritz values (possible ghost): {:?}", w);
    }
}

#[test]
fn happy_breakdown_on_invariant_subspace() {
    // Starting from an eigenvector gives an immediate happy breakdown:
    // the Krylov subspace is 1-dimensional and A-invariant.
    let n = 5;
    let a = DenseMatrix::from_fn(n, |i, j| if i == j { i as f64 + 1.0 } else { 0.0 });
    let mut v0 = vec![0.0; n];
    v0[2] = 1.0; // exact eigenvector for eigenvalue 3.0

    let result = Arnoldi::new(4, 1e-13).run(&a, &v0).unwrap();
    assert!(result.breakdown);
    assert_eq!(result.steps, 1);
    assert_relative_eq!(result.h[0][0], 3.0, epsilon = 1e-12);
}

#[test]
fn dimension_mismatch_is_rejected() {
    let a = DenseMatrix::from_fn(4, |i, j| if i == j { 1.0 } else { 0.0 });
    let v0 = vec![1.0, 0.0, 0.0]; // wrong length
    let err = Arnoldi::new(2, 1e-12).run(&a, &v0).unwrap_err();
    assert!(matches!(err, krylov_ds::KrylovError::DimensionMismatch { .. }));
}

#[test]
fn subspace_too_large_is_rejected() {
    let a = DenseMatrix::from_fn(3, |i, j| if i == j { 1.0 } else { 0.0 });
    let v0 = vec![1.0, 0.0, 0.0];
    let err = Lanczos::new(10, 1e-12, Reorthogonalization::Full).run(&a, &v0).unwrap_err();
    assert!(matches!(err, krylov_ds::KrylovError::SubspaceTooLarge { .. }));
}

#[test]
fn csr_matrix_matches_dense_equivalent() {
    let n = 5;
    let triplets = vec![
        (0usize, 0usize, 4.0),
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, 3.0),
        (1, 2, 1.0),
        (2, 1, 1.0),
        (2, 2, 5.0),
        (2, 3, 1.0),
        (3, 2, 1.0),
        (3, 3, 2.0),
        (3, 4, 1.0),
        (4, 3, 1.0),
        (4, 4, 6.0),
    ];
    let sparse = CsrMatrix::from_triplets(n, &triplets);
    let mut dense_data = vec![0.0; n * n];
    for &(r, c, v) in &triplets {
        dense_data[r * n + c] = v;
    }
    let dense = DenseMatrix::new(n, dense_data);

    let x: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    let mut y_sparse = vec![0.0; n];
    let mut y_dense = vec![0.0; n];
    sparse.apply(&x, &mut y_sparse);
    dense.apply(&x, &mut y_dense);

    for (s, d) in y_sparse.iter().zip(y_dense.iter()) {
        assert_relative_eq!(s, d, epsilon = 1e-12);
    }

    // And Lanczos on the sparse operator should match the dense one.
    let v0: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.2).collect();
    let r_sparse = Lanczos::new(n, 1e-13, Reorthogonalization::Full).run(&sparse, &v0).unwrap();
    let r_dense = Lanczos::new(n, 1e-13, Reorthogonalization::Full).run(&dense, &v0).unwrap();
    for (a, b) in r_sparse.alpha.iter().zip(r_dense.alpha.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-10);
    }
}
