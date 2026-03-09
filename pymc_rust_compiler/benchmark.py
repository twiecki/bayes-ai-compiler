"""Benchmark: compare PyMC (nutpie) vs AI-compiled Rust sampler."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm


def benchmark_nutpie(model: pm.Model, draws: int = 2000, tune: int = 1000, chains: int = 4) -> dict:
    """Benchmark PyMC sampling with nutpie backend."""
    print(f"  nutpie: {chains} chains x {draws} draws...")
    start = time.time()
    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        nuts_sampler="nutpie",
        model=model,
        random_seed=42,
        progressbar=False,
    )
    elapsed = time.time() - start
    throughput = (chains * draws) / elapsed

    return {
        "backend": "nutpie",
        "elapsed_s": elapsed,
        "throughput": throughput,
        "idata": idata,
    }


def benchmark_rust(build_dir: str | Path, draws: int = 2000, tune: int = 1000, chains: int = 4) -> dict:
    """Benchmark the AI-compiled Rust sampler."""
    build_dir = Path(build_dir)
    binary = build_dir / "target" / "release" / "sample"

    if not binary.exists():
        # Build with the sampler main
        print("  Building Rust sampler...")
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "sample"],
            cwd=build_dir,
            capture_output=True,
            check=True,
        )

    print(f"  Rust: {chains} chains x {draws} draws...")
    start = time.time()
    result = subprocess.run(
        [str(binary)],
        cwd=build_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return {"backend": "rust", "elapsed_s": elapsed, "error": result.stderr}

    throughput = (chains * draws) / elapsed

    return {
        "backend": "rust-ai",
        "elapsed_s": elapsed,
        "throughput": throughput,
        "output": result.stdout,
    }


def print_comparison(nutpie_result: dict, rust_result: dict):
    """Print a nice comparison table."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n{'Backend':<20} {'Time (s)':<12} {'Draws/sec':<12} {'Speedup':<10}")
    print("-" * 54)

    nt = nutpie_result["elapsed_s"]
    print(f"{'nutpie':<20} {nt:<12.2f} {nutpie_result['throughput']:<12.0f} {'1.00x':<10}")

    if "error" not in rust_result:
        rt = rust_result["elapsed_s"]
        speedup = nt / rt
        print(f"{'rust-ai':<20} {rt:<12.2f} {rust_result['throughput']:<12.0f} {speedup:<10.2f}x")
    else:
        print(f"{'rust-ai':<20} {'FAILED':<12}")

    print()
