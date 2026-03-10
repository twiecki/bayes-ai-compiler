# Skill: ZeroSumNormal Models

This model uses ZeroSumNormal distributions with ZeroSumTransform. The key challenge
is implementing the forward transform (unconstrained → constrained) and backpropagating
gradients through it correctly.

## How ZeroSumTransform Works

ZeroSumNormal(sigma, shape=(K,)) has K-1 unconstrained parameters that map to K
constrained values summing to zero.

**Forward transform** (K-1 unconstrained → K constrained):
```
n = K  (full dimension size)
sum_x = sum of unconstrained elements
norm = sum_x / (sqrt(n) + n)
fill = norm - sum_x / sqrt(n)    // this is the "extra" element

constrained[i] = unconstrained[i] - norm    for i = 0..K-1
constrained[K-1] = fill - norm              // last element
```

The result sums to zero: sum(constrained) = 0.

## Multi-axis ZeroSumTransform

For tensors with `n_zerosum_axes > 1`, the transform is applied sequentially to each
zero-sum axis, innermost first. For example with shape=(6,7,4) and n_zerosum_axes=2:

1. First extend axis -2 (middle): (6, **6**, 3) → (6, **7**, 3)
2. Then extend axis -1 (last): (6, 7, **3**) → (6, 7, **4**)

Each step applies the same forward transform formula along the relevant axis.

## Gradient Backpropagation

Reverse the transform order. If forward was axis -2 then axis -1:
- First backprop through axis -1 transform
- Then backprop through axis -2 transform

**Backprop formula** (K constrained gradients → K-1 unconstrained gradients):
```
n = K  (full dimension size)
sum_grad = sum of constrained gradients for elements 0..K-2
grad_fill = constrained gradient for element K-1

unconstrained_grad[i] = constrained_grad[i] - sum_grad / (sqrt(n) + n) - grad_fill / sqrt(n)
```

## ZeroSumNormal Log-density

The logp for ZeroSumNormal uses ONLY the unconstrained elements (no Jacobian needed,
it's baked into the distribution):
```
n_unc = number of unconstrained elements
logp = n_unc * (-0.5*log(2π) - log(sigma)) - 0.5 * sum(x_unc²) / sigma²
```

Gradient w.r.t. unconstrained element x_i:
```
d(logp)/d(x_i) = -x_i / sigma²
```

Gradient w.r.t. log(sigma) from ZeroSumNormal term:
```
d(logp)/d(log_sigma) = (-n_unc + sum(x_unc²) / sigma²) * sigma
```

## Pre-allocation for Effect Arrays

For small dimensions (< 64 total elements), use stack-allocated fixed arrays:
```rust
let mut store_effect_full = [0.0; 6];  // NOT vec![0.0; 6]
let mut grad_store_effect = [0.0; 6];
```

For larger tensors, pre-allocate in the struct:
```rust
pub struct GeneratedLogp {
    interaction_temp: Vec<f64>,  // flattened 3D array, reused
    interaction_full: Vec<f64>,
    grad_interaction: Vec<f64>,
}
```

Use flat indexing for multi-dimensional arrays: `arr[i * D2 * D3 + j * D3 + k]`
instead of `Vec<Vec<Vec<f64>>>` (which allocates per row).
