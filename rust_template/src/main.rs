mod generated;

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use anyhow::Result;
use nuts_rs::{
    CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, LogpError, Model,
    Sampler, SamplerWaitResult,
};
use rand::{Rng, RngExt};

use generated::{GeneratedLogp, N_PARAMS};

struct GenericModel<F: CpuLogpFunc + Clone + Send + Sync + 'static> {
    math: CpuMath<F>,
}

impl<F: CpuLogpFunc + Clone + Send + Sync + 'static> Model for GenericModel<F> {
    type Math<'model> = CpuMath<F> where Self: 'model;

    fn math<R: Rng + ?Sized>(&self, _rng: &mut R) -> Result<Self::Math<'_>> {
        Ok(self.math.clone())
    }

    fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()> {
        for p in position.iter_mut() {
            *p = rng.random_range(-0.5..0.5);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let num_chains: usize = 4;
    let num_tune = 1000;
    let num_draws = 2000;

    let mut settings = DiagGradNutsSettings::default();
    settings.num_chains = num_chains;
    settings.num_tune = num_tune;
    settings.num_draws = num_draws;
    settings.seed = 42;

    let model = GenericModel {
        math: CpuMath::new(GeneratedLogp),
    };

    println!("Sampling: {} chains x {} draws (+ {} warmup), {} params",
        num_chains, num_draws, num_tune, N_PARAMS);

    let start = Instant::now();
    let mut sampler = Some(Sampler::new(model, settings, (), num_chains, None)?);

    while let Some(s) = sampler.take() {
        match s.wait_timeout(Duration::from_millis(500)) {
            SamplerWaitResult::Trace(_) => {
                let elapsed = start.elapsed();
                println!("Completed in {:.2?}", elapsed);
                println!("Throughput: {:.0} draws/sec",
                    (num_chains * num_draws) as f64 / elapsed.as_secs_f64());
                break;
            }
            SamplerWaitResult::Timeout(s) => { sampler = Some(s); }
            SamplerWaitResult::Err(err, _) => return Err(err),
        }
    }

    Ok(())
}
