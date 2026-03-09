use std::collections::HashMap;
use std::simd::prelude::*;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use crate::data::*;

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("Recoverable: {0}")]
    Recoverable(String),
}

impl LogpError for SampleError {
    fn is_recoverable(&self) -> bool { true }
}

pub const N_PARAMS: usize = 12;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone)]
pub struct GeneratedLogp;

impl HasDims for GeneratedLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("param".to_string(), N_PARAMS as u64)])
    }
}

impl CpuLogpFunc for GeneratedLogp {
    type LogpError = SampleError;
    type FlowParameters = ();
    type ExpandedVector = Draw;

    fn dim(&self) -> usize { N_PARAMS }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SampleError> {
        let mu_a = position[0];
        let log_sigma_a = position[1];
        let a_offset = &position[2..10];
        let b = position[10];
        let log_sigma_y = position[11];

        let sigma_a = log_sigma_a.exp();
        let sigma_y = log_sigma_y.exp();

        const LN_2PI: f64 = 1.8378770664093453;

        let mut logp = 0.0;
        gradient.fill(0.0);

        // Priors (scalar, same as before)
        let mu_a_normalized = mu_a * 0.1;
        logp += -0.5 * LN_2PI - 10.0_f64.ln() - 0.5 * mu_a_normalized * mu_a_normalized;
        gradient[0] += -mu_a * 0.01;

        let sigma_a_normalized = sigma_a * 0.2;
        logp += 2.0_f64.ln() - 0.5 * LN_2PI - 5.0_f64.ln() - 0.5 * sigma_a_normalized * sigma_a_normalized + log_sigma_a;
        gradient[1] += -sigma_a * sigma_a * 0.04 + 1.0;

        for i in 0..8 {
            logp += -0.5 * LN_2PI - 0.5 * a_offset[i] * a_offset[i];
            gradient[2 + i] += -a_offset[i];
        }

        let b_normalized = b * 0.1;
        logp += -0.5 * LN_2PI - 10.0_f64.ln() - 0.5 * b_normalized * b_normalized;
        gradient[10] += -b * 0.01;

        let sigma_y_normalized = sigma_y * 0.2;
        logp += 2.0_f64.ln() - 0.5 * LN_2PI - 5.0_f64.ln() - 0.5 * sigma_y_normalized * sigma_y_normalized + log_sigma_y;
        gradient[11] += -sigma_y * sigma_y * 0.04 + 1.0;

        // Precompute a = mu_a + sigma_a * a_offset
        let mut a = [0.0; 8];
        for i in 0..8 {
            a[i] = mu_a + sigma_a * a_offset[i];
        }

        let inv_sigma_y = 1.0 / sigma_y;
        let inv_sigma_y_sq = inv_sigma_y * inv_sigma_y;
        let log_norm = -0.5 * LN_2PI - log_sigma_y;

        // Precompute mu_i for all observations (scatter a[group] + b*x into flat array)
        let mut mu_flat = [0.0f64; 256]; // enough for 170
        for i in 0..Y_N {
            let g = X_1_DATA[i] as usize;
            mu_flat[i] = a[g] + b * X_0_DATA[i];
        }

        // SIMD loop over observations
        const LANES: usize = 4;
        let chunks = Y_N / LANES;
        let remainder_start = chunks * LANES;

        let v_inv_sigma_y_sq = f64x4::splat(inv_sigma_y_sq);
        let v_log_norm = f64x4::splat(log_norm);
        let v_half = f64x4::splat(0.5);
        let v_one = f64x4::splat(1.0);
        let v_sigma_a = f64x4::splat(sigma_a);

        let mut v_logp = f64x4::splat(0.0);
        let mut v_grad_mu_a = f64x4::splat(0.0);
        let mut v_grad_b = f64x4::splat(0.0);
        let mut v_grad_log_sigma_y = f64x4::splat(0.0);

        // Group gradient accumulators (scalar, because of scatter)
        let mut grad_a_offset = [0.0f64; 8];

        for c in 0..chunks {
            let base = c * LANES;
            let v_y = f64x4::from_slice(&Y_DATA[base..]);
            let v_x = f64x4::from_slice(&X_0_DATA[base..]);
            let v_mu = f64x4::from_slice(&mu_flat[base..]);

            let v_residual = v_y - v_mu;
            let v_res_sq_scaled = v_residual * v_residual * v_inv_sigma_y_sq;

            v_logp += v_log_norm - v_half * v_res_sq_scaled;

            let v_scaled_res = v_residual * v_inv_sigma_y_sq;
            v_grad_mu_a += v_scaled_res;
            v_grad_b += v_scaled_res * v_x;
            v_grad_log_sigma_y += v_res_sq_scaled - v_one;

            // Group gradients must stay scalar (scatter pattern)
            for lane in 0..LANES {
                let i = base + lane;
                let g = X_1_DATA[i] as usize;
                grad_a_offset[g] += v_scaled_res.as_array()[lane] * sigma_a;
            }
        }

        logp += v_logp.reduce_sum();
        let mut grad_mu_a_acc = v_grad_mu_a.reduce_sum();
        let mut grad_b_acc = v_grad_b.reduce_sum();
        let mut grad_log_sigma_y_acc = v_grad_log_sigma_y.reduce_sum();

        // Scalar remainder
        for i in remainder_start..Y_N {
            let g = X_1_DATA[i] as usize;
            let x_val = X_0_DATA[i];
            let y_val = Y_DATA[i];

            let mu_i = a[g] + b * x_val;
            let residual = y_val - mu_i;
            let residual_scaled = residual * inv_sigma_y_sq;

            logp += log_norm - 0.5 * residual * residual * inv_sigma_y_sq;

            grad_mu_a_acc += residual_scaled;
            grad_b_acc += residual_scaled * x_val;
            grad_log_sigma_y_acc += residual * residual * inv_sigma_y_sq - 1.0;
            grad_a_offset[g] += residual_scaled * sigma_a;
        }

        gradient[0] += grad_mu_a_acc;
        gradient[10] += grad_b_acc;
        gradient[11] += grad_log_sigma_y_acc;

        let mut grad_log_sigma_a = 0.0;
        for i in 0..8 {
            grad_log_sigma_a += grad_a_offset[i] * a_offset[i];
            gradient[2 + i] += grad_a_offset[i];
        }
        gradient[1] += grad_log_sigma_a;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}
