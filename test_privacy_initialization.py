import importlib
import io
import itertools
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ENV_V3 as env


class PrivacyInitializationTests(unittest.TestCase):
    def test_initial_anchor_repair_respects_source_limits(self):
        origin=np.array([.1,.1,.11875,.68125])
        policy=np.array([[0,.25,.25,.25,.25],[7,.25,.25,.25,.25]])
        self.assertFalse(env.check_slo_violation(origin,policy,60))
        for limit in (0,10,20,40,60,80,100):
            fixed=env.feasible_initial_policy(policy,origin,limit)
            self.assertTrue(env.check_slo_violation(origin,fixed,limit))
            np.testing.assert_allclose(fixed[:,1:].sum(axis=1),1)
            np.testing.assert_array_equal(fixed[:,0],policy[:,0])
            np.testing.assert_array_equal(policy[:,1:],np.full((2,4),.25))
        np.testing.assert_array_equal(env.feasible_initial_policy(policy,origin,100),policy)

    def test_all_frameworks_initialize_under_strict_privacy(self):
        origin=np.array([.1,.1,.11875,.68125])
        entries=[('SOFA','sofa_parallel'),('NSGA3','nsga'),('SWFAGA','swfaga'),('Kimchi','kimchi'),('Greencourier','greencourier')]
        def evaluate(*args):
            # Reject infeasible anchors before fitting the scaler.
            if not env.check_slo_violation(args[3],args[4],args[7]):
                return [100,float('inf'),float('inf'),float('inf')],np.zeros((3,4,900))
            return [1.,2.,3.,4.],np.zeros((3,4,900))
        for module_name,function in entries:
            module=importlib.import_module(module_name)
            for limit in (0,20,40,60,80):
                with self.subTest(framework=module_name,limit=limit), tempfile.TemporaryDirectory() as directory, \
                     patch.object(module,'simulation',side_effect=evaluate) as simulate, \
                     patch.object(module,'Pool',return_value=MagicMock()), \
                     patch.object(module.time,'time',side_effect=itertools.count(0,1000)),redirect_stdout(io.StringIO()):
                    counts=np.ones((1,16));counts[0,0]=15
                    result=getattr(module,function)(dir_name=directory+'/',lut_list=[[[1,1,0,0,0]]],
                        func_distribution_arr=counts,leftover_resource_time_arr=np.zeros((3,4,900)),
                        req_distribution_arr=np.array([[0,*origin]]),epoch=95,
                        original_req_distribution_arr=origin,delay_factor=20,slo_constraint=limit,
                        decision_time=.01,seed=1)
                    self.assertIsNotNone(result)
                    for call in simulate.call_args_list:
                        self.assertTrue(env.check_slo_violation(origin,call.args[4],limit))

    def test_infeasible_scores_do_not_reach_scaler(self):
        scaler=MinMaxScaler().fit([[0]*4,[100]*4])
        for name in ('SOFA','NSGA3','SWFAGA','Kimchi','Greencourier'):
            self.assertEqual(importlib.import_module(name).local_fit([100,np.inf,np.inf,np.inf],scaler),np.inf)
        values,designs=env.finite_population([[1,2,3,4],[100,np.inf,np.inf,np.inf]],['valid','invalid'])
        self.assertEqual(values,[[1,2,3,4]])
        self.assertEqual(designs,['valid'])


if __name__=='__main__':unittest.main()
