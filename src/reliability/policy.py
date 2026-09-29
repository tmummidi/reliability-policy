"""Age replacement economics in repeated independent renewal cycles."""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def expected_cost_rate(posterior, age, preventive_cost=100, failure_cost=600):
    if age<=0 or preventive_cost<0 or failure_cost<=0:
        raise ValueError("Invalid replacement age or cost")
    if np.isinf(age):
        mean=posterior.mean_lifetime()
        if not np.isfinite(mean):
            raise ValueError("Posterior predictive mean lifetime is infinite")
        return failure_cost/mean
    survival=float(posterior.survival(age))
    uptime=quad(lambda t:float(posterior.survival(t)),0,age,epsabs=1e-7)[0]
    return (failure_cost*(1-survival)+preventive_cost*survival)/uptime


def select_age(posterior,ages,preventive_cost=100,failure_cost=600):
    candidates=list(ages)+[np.inf]
    costs=[expected_cost_rate(posterior,a,preventive_cost,failure_cost) for a in candidates]
    i=int(np.argmin(costs))
    return candidates[i],costs[i],[
        {"age":None if np.isinf(a) else float(a),"expected_cost_rate":float(c)}
        for a,c in zip(candidates,costs)]


def evaluate_paired(lifetimes,baseline_age,proposed_age,preventive_cost=100,failure_cost=600):
    life=np.asarray(lifetimes,float)
    if len(life)<2 or not np.isfinite(life).all() or (life<=0).any():
        raise ValueError("At least two positive finite evaluation lifetimes required")
    if min(baseline_age,proposed_age)<=0:
        raise ValueError("Replacement ages must be positive")
    results=[];influences=[]
    for age in (baseline_age,proposed_age):
        times=np.minimum(life,age)
        costs=np.where(life<=age,failure_cost,preventive_cost)
        rate=float(costs.sum()/times.sum())
        influences.append((costs-rate*times)/times.mean())
        results.append({"cost_per_time":rate,"mean_cycle_length":float(times.mean()),
                        "preventive_fraction":float(np.mean(life>age))})
    difference=results[0]["cost_per_time"]-results[1]["cost_per_time"]
    se=float(np.std(influences[0]-influences[1],ddof=1)/np.sqrt(len(life)))
    half=float(norm.ppf(.975)*se)
    return {"baseline":results[0],"proposed":results[1],
            "baseline_minus_proposed":{"mean":difference,"lower_95":difference-half,
                 "upper_95":difference+half,"n":len(life),
                 "method":"paired delta-method interval for two renewal reward ratios"}}
