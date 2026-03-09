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

pub const N_PARAMS: usize = 3;

const LN_2PI: f64 = 1.8378770664093453;

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
        let alpha = position[0];
        let beta = position[1];
        let log_sigma = position[2];

        let sigma = log_sigma.exp();

        gradient[0] = 0.0;
        gradient[1] = 0.0;
        gradient[2] = 0.0;

        let mut logp = 0.0;

        // Prior for alpha ~ Normal(0, 10)
        let alpha_scaled = alpha * 0.1;
        logp += -0.5 * LN_2PI - 10.0f64.ln() - 0.5 * alpha_scaled * alpha_scaled;
        gradient[0] += -alpha * 0.01;

        // Prior for beta ~ Normal(0, 10)
        let beta_scaled = beta * 0.1;
        logp += -0.5 * LN_2PI - 10.0f64.ln() - 0.5 * beta_scaled * beta_scaled;
        gradient[1] += -beta * 0.01;

        // Prior for sigma ~ HalfNormal(5) with LogTransform
        if sigma <= 0.0 {
            return Err(SampleError::Recoverable("sigma must be positive".to_string()));
        }
        let sigma_scaled = sigma * 0.2;
        logp += 2.0f64.ln() - 0.5 * LN_2PI - 5.0f64.ln() - 0.5 * sigma_scaled * sigma_scaled + log_sigma;
        gradient[2] += -0.04 * sigma * sigma + 1.0;

        // Likelihood with explicit SIMD (f64x4)
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let log_norm_const = -0.5 * LN_2PI - log_sigma;

        const LANES: usize = 4;
        let chunks = Y_N / LANES;
        let remainder_start = chunks * LANES;

        let v_alpha = f64x4::splat(alpha);
        let v_beta = f64x4::splat(beta);
        let v_inv_sigma_sq = f64x4::splat(inv_sigma_sq);
        let v_log_norm = f64x4::splat(log_norm_const);
        let v_half = f64x4::splat(0.5);
        let v_one = f64x4::splat(1.0);

        let mut v_logp = f64x4::splat(0.0);
        let mut v_grad_alpha = f64x4::splat(0.0);
        let mut v_grad_beta = f64x4::splat(0.0);
        let mut v_grad_log_sigma = f64x4::splat(0.0);

        for c in 0..chunks {
            let base = c * LANES;
            let v_y = f64x4::from_slice(&Y_DATA[base..]);
            let v_x = f64x4::from_slice(&X_0_DATA[base..]);

            let v_mu = v_alpha + v_beta * v_x;
            let v_residual = v_y - v_mu;
            let v_res_sq_scaled = v_residual * v_residual * v_inv_sigma_sq;

            v_logp += v_log_norm - v_half * v_res_sq_scaled;

            let v_scaled_res = v_residual * v_inv_sigma_sq;
            v_grad_alpha += v_scaled_res;
            v_grad_beta += v_scaled_res * v_x;
            v_grad_log_sigma += v_res_sq_scaled - v_one;
        }

        logp += v_logp.reduce_sum();
        let mut grad_alpha_acc = v_grad_alpha.reduce_sum();
        let mut grad_beta_acc = v_grad_beta.reduce_sum();
        let mut grad_log_sigma_acc = v_grad_log_sigma.reduce_sum();

        // Scalar remainder
        for i in remainder_start..Y_N {
            let y_i = Y_DATA[i];
            let x_i = X_0_DATA[i];
            let mu_i = alpha + beta * x_i;
            let residual = y_i - mu_i;
            let residual_scaled = residual * inv_sigma;

            logp += log_norm_const - 0.5 * residual_scaled * residual_scaled;

            let scaled_residual = residual * inv_sigma_sq;
            grad_alpha_acc += scaled_residual;
            grad_beta_acc += scaled_residual * x_i;
            grad_log_sigma_acc += residual_scaled * residual_scaled - 1.0;
        }

        gradient[0] += grad_alpha_acc;
        gradient[1] += grad_beta_acc;
        gradient[2] += grad_log_sigma_acc;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}
