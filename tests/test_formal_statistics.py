import unittest
from deepvision_vla.diagnostics.formal_statistics import build_paired_panel,paired_bootstrap,summarize_paired_panel


class PairedStatisticsTests(unittest.TestCase):
    def test_negative_improvement_is_retained_without_changing_cohort(self):
        records={'a':{'baseline':2.,'points':{'L0':1.,'L30':4.}},
                 'b':{'baseline':4.,'points':{'L0':2.,'L30':5.}}}
        panel=build_paired_panel(records,['blocked','a','b'],['a','b'],['L0','L30'])
        self.assertEqual(panel['signed_baseline_differences'],[[-1.,2.],[-2.,1.]])
        self.assertEqual(panel['annotation_or_protocol_blocked_source_ids'],['blocked'])
        result=summarize_paired_panel(panel)
        self.assertEqual(result['signed_baseline_difference']['means'],[-1.5,1.5])
        self.assertEqual(result['status'],'complete_metric_panel')

    def test_missing_is_not_zero_and_partial_source_not_used_on_other_points(self):
        records={'a':{'baseline':1.,'points':{'x':2.,'y':None}},
                 'b':{'baseline':0.,'points':{'x':0.,'y':0.}}}
        panel=build_paired_panel(records,['a','b','c'],['a','b','c'],['x','y'])
        self.assertEqual(panel['paired_source_ids'],['b'])
        self.assertEqual(len(panel['excluded_qualified_sources']),2)
        result=summarize_paired_panel(panel)
        self.assertEqual(result['absolute_error']['means'],[0.,0.])
        self.assertEqual(result['absolute_error']['lower'],[None,None])
        self.assertEqual(result['status'],'incomplete_metric_coverage')

    def test_alias_columns_are_paired_not_independent_replicates(self):
        # Identical masks at different nominal ratios are perfectly correlated.
        # Paired resampling must preserve that relation in every draw/interval.
        result=paired_bootstrap([[0.,0.],[2.,2.],[10.,10.]],resamples=2000)
        self.assertEqual(result['n_sources'],3)
        self.assertEqual(result['lower'][0],result['lower'][1])
        self.assertEqual(result['upper'][0],result['upper'][1])
        self.assertEqual(result['interval_scope'],'pointwise_percentile_95_not_a_simultaneous_curve_band')
        self.assertEqual(paired_bootstrap([[0.],[2.],[10.]]),paired_bootstrap([[0.],[2.],[10.]]))

    def test_frozen_order_duplicates_and_unqualified_outputs_rejected(self):
        cases=[({'a':{}},['a','a'],['a'],['x']),
               ({'a':{}},['a','b'],['b','a'],['x']),
               ({'b':{}},['a','b'],['a'],['x']),
               ({},['a'],['a'],['x','x'])]
        for args in cases:
            with self.subTest(args=args),self.assertRaises(ValueError):build_paired_panel(*args)

    def test_nonfinite_boolean_and_negative_raw_error_are_missing_evidence(self):
        for invalid in (float('nan'),float('inf'),False,-1.):
            with self.subTest(value=invalid):
                p=build_paired_panel({'a':{'baseline':1.,'points':{'x':invalid}}},['a'],['a'],['x'])
                self.assertEqual(p['paired_sources'],0)
                self.assertEqual(summarize_paired_panel(p)['absolute_error']['means'],[None])

    def test_empty_source_matrix_has_no_synthetic_mean_or_interval(self):
        import numpy as np
        self.assertEqual(paired_bootstrap(np.empty((0,2)))['means'],[None,None])
        for bad in ([[float('nan')]],[]):
            with self.assertRaises(ValueError):paired_bootstrap(bad)

    def test_saved_summary_preserves_full_requested_qualified_and_blocked_cohorts(self):
        records={'a':{'baseline':1.,'points':{'x':1.}},
                 'b':{'baseline':1.,'points':{'x':None}}}
        panel=build_paired_panel(records,['blocked','a','b'],['a','b'],['x'])
        result=summarize_paired_panel(panel)
        self.assertEqual(result['requested_source_ids'],['blocked','a','b'])
        self.assertEqual(result['qualified_source_ids'],['a','b'])
        self.assertEqual(result['annotation_or_protocol_blocked_source_ids'],['blocked'])
        self.assertEqual(result['source_ids'],['a'])
        self.assertEqual(result['excluded_qualified_sources'][0]['sample_id'],'b')


if __name__=='__main__':unittest.main()
