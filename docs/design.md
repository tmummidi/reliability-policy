# Model and release design

## User and decision

Choose preventive replacement age from censored asset lifetimes. Intended users are analysts and operations researchers
evaluating this decision before committing operational resources.

## Formulation

~~~text
With event indicator d, Weibull shape k and scale eta:
log likelihood = sum[d*(log(k)-log(eta)+(k-1)*log(t/eta)) - (t/eta)^k].
Fit k and eta by bounded maximum likelihood, using analytical gradients.

For a fixed shape k, parameterize S(t)=exp(-lambda*t^k).
Gamma(alpha,beta-rate) prior updates to:
alpha' = alpha + sum(d); beta' = beta + sum(t^k).
Posterior predictive survival is [beta'/(beta'+t^k)]^alpha'.

For replacement age a, long-run expected cost rate is:
[C_failure*(1-S(a)) + C_preventive*S(a)] / integral_0^a S(t) dt.
Compare a predeclared grid and run-to-failure. Evaluate on independent
fully observed synthetic renewal cycles using total cost / total cycle time.
~~~

## Data and provenance

Synthetic censored cohorts with separately generated fully observed evaluation lifetimes. The generator is the authoritative data source. Parameters
represent hypothetical examples, not estimates from any employer. Seeds,
configuration hashes and package versions are included with executed results.
No pre-existing code from a tutorial or employer is used in the new model.

## Architecture

survival.py implements censored likelihood, Kaplan-Meier and conditional conjugate updating. policy.py integrates predictive survival and evaluates renewal costs. cli.py separates fitting, likelihood evaluation and policy evaluation data.

Dependencies are intentionally small. NumPy handles arrays; SciPy provides
numerical/statistical routines and the free HiGHS solver where applicable.
Distributed infrastructure is not justified by these measurements.

## Evaluation and acceptance

Baseline: Fixed replacement interval.
Censoring stress, declining hazard failure case, paired renewal cost rate interval.
Meaningful tests use analytical results, enumeration, input constraints or
conservation laws. Confidence intervals and primal feasibility are reported,
not inferred from successful process exit alone.

The first release is accepted when installation into a clean environment,
unit tests and an installed CLI example all succeed. The release-validation
log records that check separately from the benchmark timings.

## Scope and limits

Independent identically distributed lifetimes, independent censoring, instantaneous replacement, as-good-as-new assets, and fixed failure/replacement costs. The Gamma update conditions on a plug-in Weibull shape; it is empirical Bayes, not a full joint Bayesian posterior. Credible intervals omit shape uncertainty.

Fewer than two failures cannot identify both Weibull parameters and raises an explicit error. Declining hazard can make run-to-failure preferable. Heavy censoring widens parameter uncertainty. A grid optimum is only optimal over the declared age candidates.

## Resource budget

CPU only; no neural models. The published benchmark is serial and uses three increasing
workloads. Very large configurations can exceed laptop memory or practical
solver time. Only the recorded configurations were executed.

## Extensions

Add a shape posterior and posterior decision regret. Model repair time, imperfect repair, left truncation and competing risks. Evaluate with a real licensed reliability cohort and censor-aware validation rather than synthetic full-life ground truth.

## Method references

- SciPy 1.13 MILP API and result status: https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.milp.html
- SimPy shared resources: https://simpy.readthedocs.io/en/stable/simpy_intro/shared_resources.html
- NIST reliability estimation with censoring: https://www.itl.nist.gov/div898/handbook/apr/section4/apr413.htm
- SciPy Gamma shape/scale parameterization: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

References describe underlying methods. Results in this repository come from
the included implementation and executable configurations.
