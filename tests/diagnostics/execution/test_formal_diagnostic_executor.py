"""Pure/PIL/NPZ/runtime-boundary tests; no engine, model or GPU is launched."""
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from deepvision_vla.diagnostics import formal_diagnostic_executor as executor
from deepvision_vla.diagnostics import formal_execution_core as core
from deepvision_vla.diagnostics import pi0_preprocessing_geometry as pi_prep


class FormalRuntimeBoundaries(unittest.TestCase):
    def test_metrics_reuse_native_outputs_with_full_chunks_and_no_borrowed_gt(self):
        profile={'identity':{'model':'pi0'}}
        prediction={'native':[[1.]*7 for _ in range(4)]};baseline={'native':[[0.]*7 for _ in range(4)]}
        result=executor.calculate_metrics(prediction,baseline,{},None,{'keep_ratio':.5,'gt_mse_allowed':False},profile)
        self.assertEqual(result['native_prediction_drift']['full_chunk_mse'],1.)
        self.assertIsNone(result['gt_mse']);self.assertIsNone(result['source_bound_dataset_reference_mse'])
        with self.assertRaises(RuntimeError):executor.calculate_metrics(prediction,baseline,{},None,{'keep_ratio':1.,'gt_mse_allowed':False},profile)
        with self.assertRaises(ValueError):executor.calculate_metrics(prediction,baseline,{},None,{'keep_ratio':.5,'gt_mse_allowed':True},profile)

    def test_cached_keep1_failure_is_protocol_error_not_metric_only_success(self):
        profile={'identity':{'model':'pi0'}};prediction={'native':[[1.]*7]*4};baseline={'native':[[0.]*7]*4}
        with self.assertRaises(RuntimeError):
            executor.evaluate_qualified_metrics(prediction,baseline,{},None,{'keep_ratio':1.,'gt_mse_allowed':False},profile)

    def source_fixture(self,root):
        import numpy as np
        from PIL import Image
        (root/'images').mkdir();(root/'arrays').mkdir()
        Image.new('RGB',(32,16),(10,20,30)).save(root/'images/source.png')
        states=np.arange(21,dtype=np.float32).reshape(3,7);states[1,6]=1.2
        np.savez(root/'arrays/source.npz',states=states,actions=np.zeros((3,7),np.float32))
        image_sha=core.file_sha(root/'images/source.png');array_sha=core.file_sha(root/'arrays/source.npz')
        original={'image_path':'images/source.png','arrays_path':'arrays/source.npz','steps':3}
        source={'sample_id':'synthetic-input-fixture','original_source':original,'image_sha256':image_sha,
            'arrays_sha256':array_sha,'image_size':[32,16],'anchor':1,'instruction':'source fixture'}
        native={'native_state8_before_normalization':states[1,:6].tolist()+[0.,float(states[1,6])]}
        audit=pi_prep.state_input_audit(states[1].tolist(),native,1,array_sha)
        layout=pi_prep.layout_from_prepared(source,[[1,3,224,224]],[1,32],[1,48],{'image_size':224,'patch_size':14})
        geometry={'actual_geometry_verified':True,'geometry_source':'actual_processor','image_sha256':image_sha,
            'arrays_sha256':array_sha,'state_input_audit':audit,'layout':asdict(layout)}
        files={path:{'bytes':(root/path).stat().st_size}for path in ('images/source.png','arrays/source.npz')}
        return source,geometry,files

    def test_actual_npz_state_anchor_is_checked_against_native_input_proof_not_frame_label(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source,geometry,files=self.source_fixture(root)
            geometry=json.loads(json.dumps(geometry))  # Real JSONL/JSON roundtrip converts every tuple to list.
            data,layout=executor.read_source(source,geometry,{'data_root':str(root),'model':'pi0'},files)
            self.assertEqual(data['image'].size,(32,16));self.assertEqual(layout.token_positions,tuple(range(256)))
            actual=pi_prep.layout_from_prepared(source,[[1,3,224,224]],[1,32],[1,48],{'image_size':224,'patch_size':14})
            self.assertEqual(layout,actual);self.assertEqual(hash(layout),hash(actual))
            for name in ('source_size','crop_xyxy','canvas_size','content_xyxy','grid_shape','token_positions'):
                self.assertIsInstance(getattr(layout,name),tuple)
            self.assertNotIn('state_frame',source)
            wrong=copy.deepcopy(geometry);wrong['state_input_audit']['measured_state7'][0]+=1
            wrong['state_input_audit']['native_state8_before_normalization'][0]+=1
            with self.assertRaises(ValueError):executor.read_source(source,wrong,{'data_root':str(root),'model':'pi0'},files)

    def test_current_source_byte_tamper_is_pre_entry_failure(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source,geometry,files=self.source_fixture(root)
            (root/'images/source.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):executor.read_source(source,geometry,{'data_root':str(root),'model':'pi0'},files)

    def test_once_marker_cannot_be_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'owned.once';permit={'permit_id':'metadata-only','scope_id':'s'}
            executor.claim_once(path,permit);original=path.read_bytes()
            with self.assertRaises(FileExistsError):executor.claim_once(path,permit)
            self.assertEqual(path.read_bytes(),original)

    def test_delegating_observer_returns_same_actual_object_and_restores_class_method(self):
        value=object()
        class ReturnControl:
            def method(self):return value
        owner=ReturnControl();original=owner.method.__func__;capture={}
        with executor.observe_return(owner,'method',capture):
            self.assertIs(owner.method(),value);self.assertIs(capture['returned_object'],value)
        self.assertNotIn('method',owner.__dict__);self.assertIs(owner.method.__func__,original)

    def test_observer_restores_instance_override_on_caller_failure_and_blocks_duplicate_body(self):
        calls=[];value=object()
        class ReturnControl:pass
        owner=ReturnControl()
        def original():calls.append(1);return value
        owner.method=original;capture={}
        with self.assertRaises(RuntimeError):
            with executor.observe_return(owner,'method',capture):
                self.assertIs(owner.method(),value);owner.method()
        self.assertEqual(calls,[1]);self.assertIs(owner.method,original)
        self.assertEqual(capture['native_calls_executed'],1);self.assertEqual(capture['blocked_duplicate_calls'],1)

    def test_observer_propagates_native_exception_and_restores_without_fake_result(self):
        class ReturnControl:
            def method(self):raise ValueError('native control error')
        owner=ReturnControl();capture={}
        with self.assertRaises(ValueError):
            with executor.observe_return(owner,'method',capture):owner.method()
        self.assertNotIn('method',owner.__dict__);self.assertNotIn('returned_object',capture)

    def test_failed_protocol_raw_values_remain_unqualified_and_nonfinite_not_zeroed(self):
        values=[float('nan')]+[0.]*6
        class ReturnControl:
            def method(self):return values
        owner=ReturnControl();capture={}
        with executor.observe_return(owner,'method',capture):owner.method()
        raw=executor.take_unqualified_return(capture,'openvla',None)
        self.assertEqual(raw['qualification'],'RAW_UNQUALIFIED');self.assertFalse(raw['verified_cache_success'])
        self.assertEqual(raw['raw_values'][0],{'nonfinite':'nan'});self.assertIsNone(raw['prediction'])
        self.assertNotIn('returned_object',capture)

    def test_actual_factory_weight_path_must_match_sealed_receipt_bytes(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);weight=root/'model.safetensors';weight.write_bytes(b'checksum-control-fixture-not-a-model')
            receipt=root/'receipt.json';receipt.write_text(json.dumps({'files':[{'path':'model.safetensors','size':weight.stat().st_size,
                'sha256':core.file_sha(weight),'verified':True}]}))
            config={'model':'pi0','asset_receipt':str(receipt),'factory_kwargs':{'checkpoint':str(root),'asset_receipt':str(receipt)}}
            self.assertEqual(executor.verify_factory_assets(config),1)
            weight.write_bytes(b'tampered')
            with self.assertRaises(ValueError):executor.verify_factory_assets(config)


if __name__=='__main__':unittest.main()
