# Reliability Policy: censored lifetimes to replacement decisions

**Decision:** Choose preventive replacement age from censored asset lifetimes.

Right-censored assets contain survival information but no observed failure time. Ignoring them biases reliability estimates and can trigger premature or late replacements.

This is a working v0.1 research engineering release, with synthetic data, a free
solver path, scientific tests, and recorded local measurements. It is independent
portfolio work. Employer systems and impact numbers are not benchmark evidence.

## Run

Python 3.10 or 3.12 is the supported CI target; the recorded local verification
used Python 3.12.14.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m reliability --output results/demo.json
python -m unittest discover -s tests -v
python -m reliability --benchmark --output results/benchmark.json
~~~

On Windows activate with .venv\Scripts\activate. The core has no paid service,
proprietary solver license, external dataset download, or GPU requirement.
Package dependencies and build tools are pinned in pyproject.toml.

To use a configuration:

~~~bash
python -m reliability --config configs/ci.json --output results/small.json
~~~

Configure cohort size, true synthetic shape/scale, censor time, costs, baseline age and evaluation cycles. The posterior API supports sequential batches without retaining raw historical observations.

## Implementation

survival.py implements censored likelihood, Kaplan-Meier and conditional conjugate updating. policy.py integrates predictive survival and evaluates renewal costs. cli.py separates fitting, likelihood evaluation and policy evaluation data.

Baseline: Fixed replacement interval. Evaluation: Censoring stress, declining hazard failure case, paired renewal cost rate interval.
The full mathematical specification is in [docs/design.md](docs/design.md).

## Executed measurements

| Workload | Runtime (s) | Peak RSS (MiB) | Measured quality |
|---|---:|---:|---|
| 500 censored training assets | 0.044 | 84.6 | cost/time 4.470 -> 4.039 |
| 5,000 censored training assets | 0.056 | 84.3 | cost/time 4.470 -> 4.039 |
| 50,000 censored training assets | 0.261 | 91.9 | cost/time 4.470 -> 4.039 |

The selected replacement age was 55.92, versus a baseline age of 100. Cost-rate reduction was 0.431 with paired 95% interval [0.359, 0.504]. The decreasing-hazard stress configuration selected run-to-failure. Repeated policy results across cohort sizes arise because the same evaluation seed and selected grid age were used.

Measurements ran on an AMD EPYC 9V74 host in a Linux container with an 8-vCPU,
8-GiB cgroup limit. These are laptop-sized workloads, not laptop hardware
measurements. One fresh process was used per scale case. Runtime includes
generation, fitting/optimization and evaluation, but excludes Python imports;
the report also records cold-process time. Peak RSS includes imports.
BLAS/OpenMP thread environment defaults were set to one; actual solver thread
utilization was not measured. No distributed scaling or parallel speedup is claimed.
Single timing observations are not latency distributions.

Raw seeds, versions, configurations, peak RSS, solver/statistical evidence and
derived throughput are in [results/benchmark.json](results/benchmark.json).
The throughput numerator is project-specific and includes only the declared work
units; it must not be compared across projects.

## Assumptions and failure cases

Independent identically distributed lifetimes, independent censoring, instantaneous replacement, as-good-as-new assets, and fixed failure/replacement costs. The Gamma update conditions on a plug-in Weibull shape; it is empirical Bayes, not a full joint Bayesian posterior. Credible intervals omit shape uncertainty.

Fewer than two failures cannot identify both Weibull parameters and raises an explicit error. Declining hazard can make run-to-failure preferable. Heavy censoring widens parameter uncertainty. A grid optimum is only optimal over the declared age candidates.

## Reproducibility and contribution

Data are generated from explicit seeds. Training and heldout scenarios are
separate; confidence intervals describe evaluation sampling variability, not
uncertainty in all model assumptions. Larger configurations in
configs/larger-unexecuted.json are **unexecuted** and are not performance claims.
No hypotheses are presented as measurements.

See [CONTRIBUTING.md](CONTRIBUTING.md), [THIRD_PARTY.md](THIRD_PARTY.md), and
[LICENSE](LICENSE). v0.1 covers the complete single-machine vertical slice above.
It is not a production deployment.
