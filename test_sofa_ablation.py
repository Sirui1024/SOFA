import importlib
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import sofa_ablation as ablation


class RecordingPool:
    def __init__(self):self.calls=[]
    def map(self, function, items):
        items=list(items); self.calls.append(function.__name__)
        if function is ablation.core.parallel_simulation:
            return [([1.,2.,3.,4.],None) for _ in items]
        return [(0,[[.5,2.,3.,4.]],[design.copy()],1.) for _,design in items]


class AblationTests(unittest.TestCase):
    def test_strict_privacy_initialization_matches_sofa_anchors(self):
        origin=np.array([.1,.1,.11875,.68125])
        calls=[]
        def simulate(*args):
            design=args[4]
            self.assertTrue(ablation.core.check_slo_violation(origin,design,20))
            calls.append(design.copy())
            return [1.,2.,3.,4.],None
        from unittest.mock import MagicMock
        import itertools
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(ablation.core,'simulation',side_effect=simulate), \
                patch.object(ablation,'SeededPool',return_value=MagicMock()), \
                patch.object(ablation.time,'monotonic',side_effect=itertools.count(0,1000)):
            counts=np.ones((1,16));counts[0,0]=15
            ablation.run_ablation(directory,[],counts,np.zeros((3,4,900)),
                np.array([[0,*origin]]),95,origin,20,20,decision_time=.01,mode='dea')
        self.assertEqual(len(calls),5)
        expected=ablation.core.feasible_initial_policy(np.array([[0,.25,.25,.25,.25]]),origin,20)
        np.testing.assert_allclose(calls[0],expected)

    def test_component_runners_save_valid_outputs(self):
        class InlinePool:
            def __init__(self, initializer=None, initargs=(), **kwargs):initializer(*initargs)
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def map(self,function,items):return list(map(function,items))
        def simulate(*args):
            shares=args[4][:,1:].mean(axis=0)
            return [float(1+shares[0]),float(1+shares[1]),float(1+shares[2]),float(1+shares[3])], np.zeros((3,4,900))
        for mode in ('dea','mls','random_ls'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory, \
                    patch.object(ablation,'SeededPool',InlinePool), \
                    patch.object(ablation.core,'simulation',side_effect=simulate):
                counts=np.ones((1,16));counts[0,0]=15
                designs,values,_=ablation.run_ablation(directory,[[[1,1,0,0,0]]],counts,
                    np.zeros((3,4,900)),np.array([[0,.25,.25,.25,.25]]),0,np.full(4,.25),20,100,
                    decision_time=.05,number_of_random_search=1,fit_frequency=1,population_size=6,seed=1,mode=mode)
                self.assertEqual(len(designs),len(values))
                self.assertLessEqual(len(values),6)
                metadata=json.loads((Path(directory)/'ablation_metadata.json').read_text())
                self.assertGreater(metadata['iterations'],0)
                self.assertTrue(np.isfinite(values).all())
                for design in designs:np.testing.assert_allclose(design[:,1:].sum(axis=1),1)

    def initial(self):
        return [[1.,2.,3.,4.],[2.,1.,3.,4.]], [np.array([[0,.25,.25,.25,.25]]) for _ in range(2)]

    def test_dea_only_has_no_local_search_or_learned_selection(self):
        values,designs=self.initial(); pool=RecordingPool()
        with patch.object(ablation.core,'GA',return_value=[designs[0].copy()]) as ga, \
             patch.object(ablation.core,'smart_brain_decision',side_effect=AssertionError('MLS called')):
            ablation.search_step('dea',values,designs,.05,pool,SimpleNamespace(),[],[])
        ga.assert_called_once()
        self.assertEqual(pool.calls,['parallel_simulation'])

    def test_mls_uses_learned_selection_and_only_local_search(self):
        values,designs=self.initial(); pool=RecordingPool(); history=[]; gains=[]
        with patch.object(ablation.core,'GA',side_effect=AssertionError('DEA called')), \
             patch.object(ablation.core,'smart_brain_decision',return_value=([0],[])) as select:
            ablation.search_step('mls',values,designs,.05,pool,SimpleNamespace(estimators_=[]),history,gains)
        select.assert_called_once()
        self.assertEqual(pool.calls,['perturb_and_simulate_10time_p'])
        self.assertEqual(len(history),1)

    def test_random_ls_has_no_dea_or_learned_selection(self):
        values,designs=self.initial();pool=RecordingPool();history=[]
        with patch.object(ablation.core,'GA',side_effect=AssertionError('DEA called')), \
             patch.object(ablation.core,'smart_brain_decision',side_effect=AssertionError('MLS called')):
            ablation.search_step('random_ls',values,designs,.05,pool,SimpleNamespace(),history,[])
        self.assertEqual(pool.calls,['perturb_and_simulate_10time_p'])
        self.assertEqual(history,[])

    def test_separate_entry_files_select_correct_component(self):
        for module,mode in [('SOFA_DEA','dea'),('SOFA_MLS','mls'),('SOFA_RandomLS','random_ls')]:
            entry=importlib.import_module(module).sofa_parallel
            self.assertIs(entry.func,ablation.run_ablation)
            self.assertEqual(entry.keywords,{'mode':mode})


if __name__=='__main__':unittest.main()
