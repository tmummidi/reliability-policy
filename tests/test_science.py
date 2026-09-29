import unittest
import numpy as np
from reliability.survival import HazardPosterior,fit_weibull,cohort,kaplan_meier,log_likelihood
from reliability.policy import select_age,evaluate_paired


class SurvivalTests(unittest.TestCase):
    def test_exponential_likelihood_closed_form(self):
        t=np.array([2.,3.,4.]);d=np.array([1,0,1])
        expected=2*np.log(.2)-.2*9
        self.assertAlmostEqual(log_likelihood(t,d,1,5),expected)

    def test_censoring_contributes_exposure_without_failure(self):
        p=HazardPosterior(2,1,10).update([2,3],[1,0])
        self.assertEqual(p.alpha,2)
        self.assertEqual(p.beta,23)
        sequential=HazardPosterior(2,1,10).update([2],[1]).update([3],[0])
        self.assertEqual(p,sequential)

    def test_mle_recovers_synthetic_parameters(self):
        t,d=cohort(20,30000,1.8,120,100)
        fit=fit_weibull(t,d)
        self.assertLess(abs(fit["shape"]-1.8),.08)
        self.assertLess(abs(fit["scale"]-120),4)

    def test_km_tied_failures_and_censoring(self):
        self.assertTrue(np.allclose(kaplan_meier([1,1,2],[1,0,1]),[(1.,2/3),(2.,0.)]))

    def test_no_failures_is_not_an_identified_mle(self):
        with self.assertRaises(ValueError):fit_weibull([1,2],[0,0])
        p=HazardPosterior(1,2,10).update([1,2],[0,0])
        self.assertEqual(p.beta,13)

    def test_predictive_survival_boundary_and_interval(self):
        p=HazardPosterior(1.8,20,3000)
        self.assertEqual(p.survival(0),1)
        self.assertTrue(np.all(np.diff(p.survival(np.arange(10)))<0))
        lo,hi=p.failure_probability_interval(10)
        self.assertTrue(0<lo<hi<1)


class PolicyTests(unittest.TestCase):
    def test_decreasing_hazard_prefers_run_to_failure(self):
        p=HazardPosterior(.7,100,100*120**.7)
        age,_,_=select_age(p,[10,50,100,500])
        self.assertTrue(np.isinf(age))

    def test_equal_policies_have_zero_paired_difference(self):
        r=evaluate_paired([10,20,30,40],25,25)
        d=r["baseline_minus_proposed"]
        self.assertEqual(d["mean"],0)
        self.assertEqual(d["lower_95"],0)

    def test_renewal_ratio_is_total_cost_over_total_time(self):
        r=evaluate_paired([10,30],20,20,100,600)
        self.assertAlmostEqual(r["baseline"]["cost_per_time"],700/30)
