import copy
import json
from pathlib import Path
import tempfile
import unittest

from deepvision_vla.diagnostics.dataset_reference_metrics import DatasetReferences
from deepvision_vla.diagnostics.formal_execution_core import file_sha


class SourceBoundReferenceContracts(unittest.TestCase):
    def fixture(self,root):
        codec={'action_definition_sha256':'a'*64,'statistics_sha256':'b'*64};cp=root/'codec.json';cp.write_text(json.dumps(codec))
        source={'sample_id':'synthetic-metadata-only','arrays_sha256':'c'*64,'anchor':3}
        row={'model':'pi0','sample_id':source['sample_id'],'arrays_sha256':source['arrays_sha256'],'anchor':3,
            'status':'verified_numeric_reference','model_native_output_codec_sha256':file_sha(cp),
            'full_trajectory_numeric_proof_sha256':'d'*64,**codec,'reference_dataset':'fixture/metadata-not-real-proof',
            'reference_revision':'f'*40,'converted_episode_index':1,'converted_frame_indices':[2,3,4,5],
            'native_reference_horizon':[[0.]*7 for _ in range(4)]}
        artifact={'schema':'verified_source_bound_dataset_references_v1','global_physical_gt_claim':False,
            'model_outcomes_used_for_reference_selection':False,'accepted_full_trajectory_numeric_proof_sha256':['d'*64],
            'references':[row]};ap=root/'references.json';ap.write_text(json.dumps(artifact))
        return source,row,artifact,cp,ap
    def test_complete_source_bound_metrics_do_not_enable_global_gt(self):
        with tempfile.TemporaryDirectory()as directory:
            source,row,artifact,cp,ap=self.fixture(Path(directory));refs=DatasetReferences(ap,file_sha(ap),cp,file_sha(cp))
            result=refs.evaluate({'native':[[1.]*7 for _ in range(4)]},source,'pi0')
            self.assertEqual(result['source_bound_dataset_reference_mse'],1.)
            self.assertFalse(result['global_physical_gt_verified'])
    def test_missing_qwen_contract_stays_null_and_does_not_borrow_pi_values(self):
        with tempfile.TemporaryDirectory()as directory:
            source,row,artifact,cp,ap=self.fixture(Path(directory));refs=DatasetReferences(ap,file_sha(ap),cp,file_sha(cp))
            result=refs.evaluate({'native':[[0.]*7]*16,'normalized':[[0.]*7]*16},source,'qwen_h16')
            self.assertIsNone(result['source_bound_dataset_reference_mse']);self.assertEqual(result['dataset_reference_status'],'missing_verified_reference')
    def test_short_window_wrong_anchor_stats_or_unaccepted_proof_yields_named_null(self):
        for field,value in (('converted_frame_indices',[2,3]),('anchor',4),('statistics_sha256','e'*64),
                            ('full_trajectory_numeric_proof_sha256','e'*64),('native_reference_horizon',[[0.]*7])):
            with tempfile.TemporaryDirectory()as directory:
                source,row,artifact,cp,ap=self.fixture(Path(directory));artifact['references'][0][field]=value;ap.write_text(json.dumps(artifact))
                refs=DatasetReferences(ap,file_sha(ap),cp,file_sha(cp));result=refs.evaluate({'native':[[0.]*7]*4},source,'pi0')
                self.assertIsNone(result['source_bound_dataset_reference_mse'])
    def test_duplicate_ambiguous_reference_is_not_silently_selected(self):
        with tempfile.TemporaryDirectory()as directory:
            source,row,artifact,cp,ap=self.fixture(Path(directory));artifact['references'].append(copy.deepcopy(row));ap.write_text(json.dumps(artifact))
            refs=DatasetReferences(ap,file_sha(ap),cp,file_sha(cp))
            result=refs.evaluate({'native':[[0.]*7]*4},source,'pi0')
            self.assertEqual(result['dataset_reference_status'],'ambiguous_duplicate_reference')
            self.assertIsNone(result['source_bound_dataset_reference_mse'])

if __name__=='__main__':unittest.main()
