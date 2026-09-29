import numpy as np
from .survival import cohort,fit_weibull,HazardPosterior,kaplan_meier,log_likelihood
from .policy import select_age,evaluate_paired
from .evidence import cli

DEFAULT=dict(n=2000,shape=1.8,scale=120.0,censor_time=100.0,evaluation_cycles=20000,
             baseline_age=100.0,preventive_cost=100.0,failure_cost=600.0,seed=20260929)
SCALES=[dict(n=n,evaluation_cycles=10000) for n in (500,5000,50000)]


def run(c):
    """Fit censored survival, update risk and evaluate a replacement policy."""
    seeds=[int(x) for x in np.random.SeedSequence(c["seed"]).generate_state(3)]
    t,d=cohort(seeds[0],c["n"],c["shape"],c["scale"],c["censor_time"])
    fit=fit_weibull(t,d)
    # Prior: one pseudo-failure and exposure of one nominal 100-time-unit life.
    prior=HazardPosterior(fit["shape"],1.0,100.0**fit["shape"])
    posterior=prior.update(t,d)
    age,expected,curve=select_age(posterior,np.geomspace(5,1000,80),c["preventive_cost"],c["failure_cost"])
    rng=np.random.default_rng(seeds[1])
    lives=c["scale"]*rng.weibull(c["shape"],c["evaluation_cycles"])
    evaluated=evaluate_paired(lives,c["baseline_age"],age,c["preventive_cost"],c["failure_cost"])
    test_t,test_d=cohort(seeds[2],max(500,c["n"]//4),c["shape"],c["scale"],c["censor_time"])
    exp_scale=float(t.sum()/d.sum())
    km=kaplan_meier(t,d)
    indices=np.unique(np.linspace(0,len(km)-1,min(40,len(km))).astype(int))
    return {"work_units":{"count":c["n"]+c["evaluation_cycles"],"unit":"fit and renewal evaluation observations"},"project":"reliability","data":"independent synthetic right-censored lifetimes",
            "seeds":{"fit":seeds[0],"policy_evaluation":seeds[1],"likelihood_evaluation":seeds[2]},
            "censor_fraction":float(1-d.mean()),"weibull_mle":fit,
            "posterior":{"shape_fixed_at_mle":posterior.shape,"alpha":posterior.alpha,"beta":posterior.beta,
                "predictive_failure_probability_30":float(1-posterior.survival(30)),
                "conditional_failure_probability_30_credible_interval":posterior.failure_probability_interval(30)},
            "kaplan_meier_sample":[km[i] for i in indices],
            "holdout_negative_log_likelihood_per_asset":{
                "weibull":-log_likelihood(test_t,test_d,fit["shape"],fit["scale"])/len(test_t),
                "exponential_baseline":-log_likelihood(test_t,test_d,1,exp_scale)/len(test_t)},
            "selected_age":None if np.isinf(age) else float(age),
            "selected_policy":"run_to_failure" if np.isinf(age) else "age_replacement",
            "predicted_cost_rate":float(expected),"age_cost_curve":curve,
            "heldout_renewal_evaluation":evaluated,
            "uncertainty_limit":"Intervals condition on fitted shape; policy evaluation uses fully observed synthetic lives, not censored empirical ground truth"}


def main():
    cli(run,DEFAULT,SCALES,"reliability")
