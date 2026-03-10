# Skill: Gaussian Process Models

This model contains a Gaussian Process. GP models require matrix operations
(Cholesky decomposition, solves, inverses) which need special handling in Rust.

## Dependencies

Add `faer = "0.24"` to Cargo.toml. faer is a pure-Rust linear algebra library
with excellent performance (comparable to LAPACK).

## Pre-allocated Struct Pattern

GP models MUST pre-allocate all matrices and scratch buffers in the struct.
Matrix allocation per logp call is catastrophic for performance.

```rust
use faer::{Mat, Par, Spec};
use faer::linalg::cholesky::llt::{factor, solve, inverse};
use faer::linalg::cholesky::llt::factor::LltParams;
use faer::dyn_stack::{MemBuffer, MemStack};

pub struct GeneratedLogp {
    k_mat: Mat<f64>,      // N×N kernel matrix (overwritten each call)
    alpha: Mat<f64>,      // N×1 for K^{-1} y
    kinv: Mat<f64>,       // N×N for inverse
    dk_dparam: Vec<f64>,  // N*N flat storage for kernel derivatives
    chol_buf: MemBuffer,  // scratch for cholesky_in_place
    inv_buf: MemBuffer,   // scratch for inverse
}

impl GeneratedLogp {
    pub fn new() -> Self {
        let chol_scratch = factor::cholesky_in_place_scratch::<f64>(
            N, Par::Seq, Spec::<LltParams, f64>::default(),
        );
        let inv_scratch = inverse::inverse_scratch::<f64>(N, Par::Seq);
        Self {
            k_mat: Mat::zeros(N, N),
            alpha: Mat::zeros(N, 1),
            kinv: Mat::zeros(N, N),
            dk_dparam: vec![0.0; N * N],
            chol_buf: MemBuffer::new(chol_scratch),
            inv_buf: MemBuffer::new(inv_scratch),
        }
    }
}
```

## Cholesky Decomposition (in-place)

```rust
// Overwrites k_mat lower triangle with L where K = L L^T
factor::cholesky_in_place(
    self.k_mat.as_mut(),
    Default::default(),
    Par::Seq,
    MemStack::new(&mut self.chol_buf),
    Spec::<LltParams, f64>::default(),
).map_err(|_| {
    SampleError::Recoverable("Cholesky failed: not positive definite".to_string())
})?;
```

## Solving K x = b (after Cholesky)

```rust
// alpha starts as y, becomes K^{-1} y in-place
for i in 0..N {
    self.alpha[(i, 0)] = Y_DATA[i];
}
solve::solve_in_place(
    self.k_mat.as_ref(),  // L from Cholesky
    self.alpha.as_mut(),
    Par::Seq,
    MemStack::new(&mut []),
);
```

## Log-determinant

```rust
// After Cholesky: log|K| = 2 * sum(log(L_ii))
let mut log_det = 0.0;
for i in 0..N {
    log_det += self.k_mat[(i, i)].ln();
}
log_det *= 2.0;
```

## Computing K^{-1} (for gradients)

```rust
inverse::inverse(
    self.kinv.as_mut(),
    self.k_mat.as_ref(),  // L from Cholesky
    Par::Seq,
    MemStack::new(&mut self.inv_buf),
);
// Note: only lower triangle is filled. Access symmetrically:
let kinv_ij = if i >= j { self.kinv[(i, j)] } else { self.kinv[(j, i)] };
```

## GP Log-likelihood

For y ~ MvNormal(0, K):
```
logp = -0.5 * (N * log(2π) + log|K| + y^T K^{-1} y)
```

## GP Gradients

For a kernel hyperparameter θ with derivative dK/dθ:
```
d(logp)/dθ = -0.5 * tr((K^{-1} - α α^T) dK/dθ)
```
where α = K^{-1} y. Compute element-wise:
```rust
for i in 0..N {
    for j in 0..N {
        let w_ij = kinv_ij - alpha_i * alpha_j;
        grad_theta += w_ij * dk_dtheta[i * N + j];
    }
}
gradient[idx] += -0.5 * grad_theta;
```

## Common Kernels

**ExpQuad (RBF):**
```
K_ij = eta^2 * exp(-0.5 * |x_i - x_j|^2 / ls^2)
dK/d(log_ls) = K_ij * |x_i - x_j|^2 / ls^2  (chain rule through log)
dK/d(log_eta) = 2 * K_ij  (chain rule through log)
```

**White noise (diagonal):**
```
K_ij += sigma^2 * delta_ij
dK/d(log_sigma) = 2 * sigma^2 * delta_ij
```

## JITTER

Always add a small jitter (1e-6) to the diagonal for numerical stability:
```rust
const JITTER: f64 = 1e-6;
// When building K:
if i == j { k[(i,j)] = kernel_ij + sigma_sq + JITTER; }
```
