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

pub const N_PARAMS: usize = 2;
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
        let mu = position[0];
        let log_sigma = position[1];
        let sigma = log_sigma.exp();

        if sigma <= 0.0 {
            return Err(SampleError::Recoverable("sigma must be positive".to_string()));
        }

        let mut logp = 0.0;
        gradient[0] = 0.0;
        gradient[1] = 0.0;

        // Prior: mu ~ Normal(0, 10)
        let mu_scaled = mu * 0.1;
        logp += -0.5 * LN_2PI - 10.0f64.ln() - 0.5 * mu_scaled * mu_scaled;
        gradient[0] += -mu * 0.01;

        // Prior: sigma ~ HalfNormal(5) with LogTransform
        let sigma_scaled = sigma * 0.2;
        logp += 2.0f64.ln() - 0.5 * LN_2PI - 5.0f64.ln() - 0.5 * sigma_scaled * sigma_scaled + log_sigma;
        gradient[1] += -sigma * sigma * 0.04 + 1.0;

        // Likelihood with SIMD
        let inv_sigma_sq = 1.0 / (sigma * sigma);
        let log_norm_term = -0.5 * LN_2PI - log_sigma;

        const LANES: usize = 4;
        let chunks = Y_N / LANES;
        let remainder_start = chunks * LANES;

        let v_mu = f64x4::splat(mu);
        let v_inv_sigma_sq = f64x4::splat(inv_sigma_sq);
        let v_log_norm = f64x4::splat(log_norm_term);
        let v_half = f64x4::splat(0.5);
        let v_one = f64x4::splat(1.0);

        let mut v_logp = f64x4::splat(0.0);
        let mut v_grad_mu = f64x4::splat(0.0);
        let mut v_grad_log_sigma = f64x4::splat(0.0);

        for c in 0..chunks {
            let base = c * LANES;
            let v_y = f64x4::from_slice(&Y_DATA[base..]);
            let v_residual = v_y - v_mu;
            let v_res_sq_scaled = v_residual * v_residual * v_inv_sigma_sq;

            v_logp += v_log_norm - v_half * v_res_sq_scaled;
            v_grad_mu += v_residual * v_inv_sigma_sq;
            v_grad_log_sigma += v_res_sq_scaled - v_one;
        }

        logp += v_logp.reduce_sum();
        let mut grad_mu_acc = v_grad_mu.reduce_sum();
        let mut grad_log_sigma_acc = v_grad_log_sigma.reduce_sum();

        for i in remainder_start..Y_N {
            let residual = Y_DATA[i] - mu;
            logp += log_norm_term - 0.5 * residual * residual * inv_sigma_sq;
            grad_mu_acc += residual * inv_sigma_sq;
            grad_log_sigma_acc += residual * residual * inv_sigma_sq - 1.0;
        }

        gradient[0] += grad_mu_acc;
        gradient[1] += grad_log_sigma_acc;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}
