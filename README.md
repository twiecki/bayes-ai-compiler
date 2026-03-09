# PyMC Rust AI Compiler

Compile PyMC models to optimized Rust via LLM. The AI doesn't just translate ops mechanically — it reasons about the full computational graph and applies optimizations a human expert would: loop fusion, memory pre-allocation, cache-friendly access patterns.

## How it works

```
PyMC Model → Extract logp graph + validation points → Claude API → Rust code → Verify → Compile
```

1. **Extract**: Read `pm.Model()` to get parameters, transforms, logp graph, and reference values
2. **Generate**: Send to Claude API which generates a complete Rust `CpuLogpFunc` implementation
3. **Verify**: Build and validate against PyMC's exact logp + gradient values
4. **Retry**: If validation fails, feed errors back to Claude (up to 3 attempts)

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install -e .
```

```python
import pymc as pm
from pymc_rust_compiler import compile_model

# Define your model
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 10)
    sigma = pm.HalfNormal("sigma", 5)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

# Compile to Rust (calls Claude API, builds, validates)
result = compile_model(model, build_dir="compiled_models/my_model")

if result.success:
    print(f"Done in {result.n_attempts} attempt(s)")
```

## Examples

```bash
# Run a single model
python examples/01_normal.py

# Full benchmark suite (Normal, LinReg, Hierarchical)
python examples/run_benchmark.py
```

## Architecture

```
pymc_rust_compiler/
├── exporter.py     # Extract everything from pm.Model()
├── compiler.py     # Claude API → Rust code → build → validate loop
└── benchmark.py    # Compare nutpie vs AI-compiled Rust
```

The key insight: `pm.Model()` already contains everything we need — parameters, transforms, shapes, logp functions. No need to overload anything. We just read it and generate optimized Rust.
