"""Right-censored Weibull likelihood and conditional conjugate updating."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import gamma


def cohort(seed, n, shape=1.8, scale=120.0, censor_time=100.0):
    if n < 1 or min(shape,scale,censor_time) <= 0:
        raise ValueError("Positive cohort parameters required")
    rng=np.random.default_rng(seed)
    lifetime=scale*rng.weibull(shape,n)
    # Administrative censoring independent of latent lifetimes.
    censor=rng.uniform(.6*censor_time,1.4*censor_time,n)
    return np.minimum(lifetime,censor), (lifetime<=censor).astype(int)


def validate(times, events):
    t=np.asarray(times,float);d=np.asarray(events)
    if t.ndim!=1 or d.shape!=t.shape or len(t)==0 or not np.isfinite(t).all() or (t<=0).any() or not np.isin(d,[0,1]).all():
        raise ValueError("Positive finite times and binary event flags required")
    return t,d


def log_likelihood(times, events, shape, scale):
    t,d=validate(times,events)
    if min(shape,scale)<=0:
        raise ValueError("Positive Weibull parameters required")
    return float(np.sum(d*(np.log(shape)-np.log(scale)+(shape-1)*np.log(t/scale))-(t/scale)**shape))


def fit_weibull(times, events):
    t,d=validate(times,events)
    if d.sum()<2:
        raise ValueError("At least two observed failures needed to estimate both Weibull parameters")
    logt=np.log(t)
    def objective(theta):
        k,scale=np.exp(theta)
        ratio=logt-np.log(scale)
        z=np.exp(k*ratio)
        loss=-np.sum(d*(np.log(k)-np.log(scale)+(k-1)*ratio)-z)
        grad=np.array([-np.sum(d*(1+k*ratio)-k*z*ratio), k*(d.sum()-z.sum())])
        return float(loss),grad
    initial=[0,np.log(t.sum()/d.sum())]
    result=minimize(objective,initial,jac=True,method="L-BFGS-B",
                    bounds=[(np.log(.15),np.log(8)),(np.log(.01),np.log(1e6))],
                    options={"maxiter":500,"ftol":1e-12})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Weibull MLE did not converge: "+result.message)
    k,scale=np.exp(result.x)
    return {"shape":float(k),"scale":float(scale),"negative_log_likelihood":float(result.fun),
            "observed_failures":int(d.sum()),"observations":len(t),"iterations":int(result.nit),
            "at_parameter_bound":bool(np.isclose(k,.15) or np.isclose(k,8) or np.isclose(scale,.01) or np.isclose(scale,1e6))}


@dataclass(frozen=True)
class HazardPosterior:
    """Gamma(shape=alpha, rate=beta) for lambda in S(t)=exp(-lambda*t^k).

    k is fixed, often a plug-in MLE. Credible intervals condition on k and do
    not include uncertainty in that fitted shape parameter.
    """
    shape: float
    alpha: float
    beta: float

    def __post_init__(self):
        if not np.isfinite([self.shape,self.alpha,self.beta]).all() or min(self.shape,self.alpha,self.beta)<=0:
            raise ValueError("Positive finite posterior parameters required")

    def update(self,times,events):
        t,d=validate(times,events)
        return HazardPosterior(self.shape,self.alpha+float(d.sum()),self.beta+float(np.sum(t**self.shape)))

    def survival(self,t):
        t=np.asarray(t,float)
        if (t<0).any():
            raise ValueError("Survival times must be nonnegative")
        return np.exp(-self.alpha*np.log1p(t**self.shape/self.beta))

    def mean_lifetime(self):
        inv=1/self.shape
        if self.alpha<=inv:
            return np.inf
        return float(np.exp(inv*np.log(self.beta)+gammaln(1+inv)+gammaln(self.alpha-inv)-gammaln(self.alpha)))

    def failure_probability_interval(self,horizon,level=.95):
        if horizon<0 or not 0<level<1:
            raise ValueError("Invalid horizon or credible level")
        lambdas=gamma.ppf([(1-level)/2,1-(1-level)/2],a=self.alpha,scale=1/self.beta)
        return (-np.expm1(-lambdas*horizon**self.shape)).tolist()


def kaplan_meier(times,events):
    t,d=validate(times,events)
    points=np.unique(t)
    survival=1.0;out=[]
    # Sort once to avoid a quadratic risk-set calculation.
    order=np.argsort(t);t=t[order];d=d[order]
    start=0
    for point in points:
        end=np.searchsorted(t,point,side="right")
        failures=d[start:end].sum()
        survival*=1-failures/(len(t)-start)
        out.append((float(point),float(survival)))
        start=end
    return out
