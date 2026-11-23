"""Pure source/input/resource/permit tests; no Torch/model/GPU execution."""
import ast
import builtins
from copy import deepcopy
import importlib
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import symtable
import tempfile
import unittest
from unittest.mock import patch

from deepvision_vla.diagnostics import paper_probe, public_surface, source_manifest
from deepvision_vla.diagnostics.predictors import finite_chunk, chunk_bits, chunk_drift, HORIZON
from deepvision_vla.diagnostics.input_payloads import prepare_sample_inputs
from deepvision_vla.diagnostics.resources import require_own_memory
from deepvision_vla.diagnostics.model_geometry import actual_visual_layout
from deepvision_vla.diagnostics.adapters.openvla_gradients import compare_native_ce, teacher_sequence, action_logit_positions
from deepvision_vla.diagnostics.adapters.qwen_h16_gradients import prediction_float32_bytes


class PublicCoreContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def valid_config_permit(self):
        config = {'schema': 'deepvision_paper_probe_config_v1', 'model': 'openvla',
                  'factory': {'checkpoint': 'weights', 'asset_receipt': 'assets.json', 'official_inventory': 'official.json'},
                  'source_manifest_sha256': 'b'*64}
        permit = {'schema': 'deepvision_paper_probe_permit_v1', 'model': 'openvla', 'config_sha256': 'a'*64,
                  'source_manifest_sha256': 'b'*64, 'physical_gpu': 2, 'exclusive_gpu': True,
                  'allow_gpu_sharing': False, 'expected_gpu_uuid': 'GPU-12345678-1234-1234-1234-123456789abc',
                  'expected_gpu_pci': '0000:62:00.0', 'budget_origin_epoch': 1000.,
                  'budget_origin_monotonic': 20., 'deadline_epoch': 2200., 'deadline_monotonic': 1220.,
                  'maximum_own_gpu_bytes': 24_000_000_000, 'allocator_gpu_bytes': 22_000_000_000,
                  'minimum_forward_free_bytes': 8_000_000_000, 'maximum_rss_bytes': 32_000_000_000,
                  'external_watchdog_required': True, 'cpu_threads': 8}
        return config, permit

    def test_native_chunks_not_truncated_and_signed_zero_not_aliased(self):
        self.assertEqual(HORIZON, {'openvla': 1, 'pi0': 4, 'qwen_h16': 16})
        full = [[0.]*7 for _ in range(16)]
        self.assertIs(finite_chunk(full, 16), full)
        with self.assertRaises(ValueError): finite_chunk(full[:1], 16)
        signed = deepcopy(full); signed[0][0] = -0.
        self.assertNotEqual(chunk_bits(full), chunk_bits(signed))
        self.assertNotEqual(prediction_float32_bytes(full), prediction_float32_bytes(signed))
        self.assertEqual(chunk_drift(full, signed)['full_chunk_mse'], 0.)

    def test_actual_qwen_processor_geometry_not_early_7x7_assumption(self):
        sample = {'image_size': [256,256], 'image_sha256': 'a'*64}
        observed = {'image_grid_thw': [[1,16,16]], 'spatial_merge_size': 2, 'patch_size': 14,
                    'visual_positions': list(range(4,68)), 'action_query_count':16,
                    'original_image_size':[256,256], 'prepared_image_size':[224,224],
                    'image_transform':'full_image_bicubic_resize_224'}
        layout = actual_visual_layout(sample, observed)
        self.assertEqual(layout.grid_shape,(8,8)); self.assertEqual(layout.token_positions,tuple(range(4,68)))
        observed['visual_positions'] = list(range(4,53))
        with self.assertRaises(ValueError): actual_visual_layout(sample, observed)

    def test_native_openvla_ce_causal_positions_and_machine_precision_only(self):
        self.assertEqual(teacher_sequence([1,2],list(range(7))),([1,2]+list(range(7)),[-100,-100]+list(range(7))))
        self.assertEqual(action_logit_positions(2,256),tuple(range(257,264)))
        self.assertTrue(compare_native_ce(1.+2**-24,1.)['within_bound'])
        self.assertFalse(compare_native_ce(1.01,1.)['within_bound'])

    def test_unknown_own_memory_is_not_zero(self):
        for info in ({'processes':[]},{'processes':[{'pid':5,'used_bytes':None}]}):
            with self.assertRaises(RuntimeError): require_own_memory(info,5)
        self.assertEqual(require_own_memory({'processes':[{'pid':5,'used_bytes':12}]},5),12)

    def test_pi_native_input_preserves_raw_gripper_and_has_no_frame_claim(self):
        import numpy as np
        from PIL import Image
        image = self.root/'image.png'; arrays = self.root/'arrays.npz'
        Image.new('RGB',(16,16)).save(image)
        values = np.array([[1,2,3,4,5,6,1.25]],dtype=np.float32)
        np.savez_compressed(arrays,states=values)
        _, state, audit = prepare_sample_inputs({'anchor':0,'instruction':'déplier','image_size':[16,16]},image,arrays)
        self.assertEqual(state,[1,2,3,4,5,6,1.25])
        self.assertEqual(audit['native_state8'],[1,2,3,4,5,6,0,1.25])
        self.assertIs(audit['physical_frame_origin_verified'],False)
        self.assertIs(audit['extra_frame_transform_applied'],False)

    def test_permit_is_exclusive_fixed_budget_and_no_implicit_factory_fields(self):
        config, permit = self.valid_config_permit()
        call=lambda:paper_probe.validate_configuration(config,permit,'a'*64,now_epoch=1100.,now_monotonic=120.)
        self.assertEqual(call(),'openvla')
        for key,value in [('allow_gpu_sharing',True),('exclusive_gpu',False),('cpu_threads',2),('physical_gpu',True),('deadline_epoch',3000.)]:
            bad=deepcopy(permit);bad[key]=value
            with self.assertRaises(ValueError):paper_probe.validate_configuration(config,bad,'a'*64,now_epoch=1100.,now_monotonic=120.)
        with self.assertRaises(ValueError):paper_probe.validate_configuration(config,permit,'a'*64,now_epoch=2300.,now_monotonic=1320.)
        config['factory']['random_fallback']=True
        with self.assertRaises(ValueError):call()

    def test_qwen_attribution_counts_fresh_unmasked_forward_separately(self):
        from deepvision_vla.diagnostics.adapters import qwen_h16_output

        probes = [{"layer": layer, "status": "complete", "baseline_comparison": {"bit_exact": True}}
                  for layer in qwen_h16_output.LAYERS]
        legacy = {"model_forward_calls": 4, "head_forward_calls": 4, "backward_calls": 4,
                  "probes_completed": 4, "probes": probes}
        fresh = {**legacy, "baseline_forward_calls": 1, "model_forward_calls": 5, "head_forward_calls": 5}
        qwen_h16_output.validate_paper_probe_counts(legacy)
        qwen_h16_output.validate_paper_probe_counts(fresh)
        with self.assertRaises(RuntimeError):
            qwen_h16_output.validate_paper_probe_counts({**fresh, "backward_calls": 5})
        with self.assertRaises(RuntimeError):
            qwen_h16_output.validate_paper_probe_counts({**legacy, "model_forward_calls": 5})

    def test_baseline_binding_tamper_or_missing_native_equivalence_rejected(self):
        sample=dict(sample_id='a'*20,image_sha256='b'*64,arrays_sha256='c'*64,anchor=7,instruction='déplier')
        binding={'schema':'deepvision_qwen_baseline_binding_v1',**sample,'baseline_sha256':'d'*64,
                 'source_manifest_sha256':'e'*64,'status':'verified','native_equivalence_verified':True}
        paper_probe.validate_baseline_binding(binding,sample,'d'*64,'e'*64)
        for key,value in [('anchor',8),('arrays_sha256','f'*64),('native_equivalence_verified',False)]:
            bad=dict(binding,**{key:value})
            with self.assertRaises(ValueError):paper_probe.validate_baseline_binding(bad,sample,'d'*64,'e'*64)

    def test_source_manifest_tamper_duplicate_traversal_never_historical_promotion(self):
        (self.root/'one.py').write_text('value=1\n',encoding='utf-8')
        manifest=source_manifest.build_source_manifest(self.root,['one.py'])
        self.assertFalse(manifest['historical_cache_equivalence_approved'])
        self.assertEqual(source_manifest.verify_source_manifest(self.root,manifest),manifest)
        for files in (['one.py','one.py'],['../escape.py'],['C:/escape.py']):
            with self.assertRaises((ValueError,FileNotFoundError)):source_manifest.build_source_manifest(self.root,files)
        (self.root/'one.py').write_text('value=2\n',encoding='utf-8')
        with self.assertRaises(ValueError):source_manifest.verify_source_manifest(self.root,manifest)

    def test_utf8_metadata_is_actual_unicode_and_checksum_bound(self):
        path=self.root/'metadata.json';path.write_text(json.dumps({'instruction':'déplier'},ensure_ascii=False),encoding='utf-8')
        self.assertEqual(paper_probe.sealed_metadata(path,source_manifest.file_sha256(path))['instruction'],'déplier')
        with self.assertRaises(ValueError):paper_probe.sealed_metadata(path,'0'*64)

    def test_public_surface_imports_no_private_modules_or_neural_runtime(self):
        src=Path(__file__).parents[1]/'src'
        audit=public_surface.verify_public_surface(src)
        self.assertGreater(audit['modules'],40)
        for name in public_surface.MODULES:importlib.import_module(name)
        self.assertNotIn('torch',sys.modules)
        self.assertFalse(any(('canary' in name or 'role_pilot' in name) for name in sys.modules if name.startswith('deepvision_vla.diagnostics.')))

    def test_release_overlay_excludes_private_operations_and_never_overwrites(self):
        src=Path(__file__).parents[1]/'src';out=self.root/'overlay'
        manifest=public_surface.stage_public_sources(src,out)
        names={r['path'] for r in manifest['files']}
        self.assertFalse(any(('canary' in n or 'role_pilot' in n or 'legacy_success_cache_export' in n) for n in names))
        self.assertFalse(manifest['historical_cache_equivalence_approved'])
        with self.assertRaises(FileExistsError):public_surface.stage_public_sources(src,out)

    def test_static_dynamic_import_alias_cannot_escape_allowlist(self):
        src=Path(__file__).parents[1]/'src'
        out=self.root/'overlay';public_surface.stage_public_sources(src,out)
        path=out/'src/deepvision_vla/diagnostics/provenance.py'
        with path.open('a',encoding='utf-8') as stream:
            stream.write('\nfrom importlib import import_module as hidden\ndef bad():\n    return hidden("private_helper")\n')
        with self.assertRaisesRegex(ValueError,'dynamic'):public_surface.verify_public_surface(out/'src')

    def test_default_entry_never_loads_model_on_validation_failure(self):
        with patch.object(paper_probe,'cpu_factory',side_effect=AssertionError('NN factory must not run')):
            with self.assertRaises((FileNotFoundError,ValueError)):
                paper_probe.run(self.root/'missing','a'*64,self.root/'permit','b'*64)

    def test_all_public_native_function_globals_are_resolved_without_execution(self):
        src=Path(__file__).parents[1]/'src'
        missing=[]
        for module in public_surface.MODULES:
            path=public_surface.module_file(src,module)
            table=symtable.symtable(path.read_text(encoding='utf-8'),str(path),'exec')
            names=set(table.get_identifiers())|set(dir(builtins))|{'__file__','__name__','__package__','__spec__','__loader__'}
            def check(block):
                for symbol in block.get_symbols():
                    if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in names:
                        missing.append((module,block.get_name(),symbol.get_name()))
                for child in block.get_children():check(child)
            check(table)
        self.assertEqual(missing,[])

    def test_prepared_examples_are_not_execution_permits(self):
        config,permit=self.valid_config_permit();permit['prepared_not_authorized']=True
        with self.assertRaisesRegex(ValueError,'Prepared'):
            paper_probe.validate_configuration(config,permit,'a'*64,now_epoch=1100.,now_monotonic=120.)

    def test_factory_receipt_seals_cannot_be_omitted_or_tampered(self):
        receipt=self.root/'receipt.json';receipt.write_text('{}',encoding='utf-8')
        official=self.root/'official.json';official.write_text('{}',encoding='utf-8')
        config={'model':'openvla','factory':{'asset_receipt':str(receipt),'official_inventory':str(official)}}
        with self.assertRaises(ValueError):paper_probe.validate_factory_metadata_seals(config)
        config['factory_metadata_seals']=[{'path':str(p),'sha256':source_manifest.file_sha256(p)} for p in (receipt,official)]
        self.assertEqual(len(paper_probe.validate_factory_metadata_seals(config)),2)
        receipt.write_text('{"changed":true}',encoding='utf-8')
        with self.assertRaises(ValueError):paper_probe.validate_factory_metadata_seals(config)

    def test_parent_must_be_the_actual_sealed_watchdog_not_just_any_live_pid(self):
        watchdog=self.root/'watchdog.py';watchdog.write_text('# reviewed parent\n',encoding='utf-8')
        parent={'pid':123,'start_ticks':456,'boot_id':'fixed-boot'}
        command=b'python\0'+str(watchdog.resolve()).encode()+b'\0'
        intent={'schema':'deepvision_probe_parent_intent_v1','parent':parent,'config_sha256':'a'*64,
                'budget_origin_epoch':1000.,'budget_origin_monotonic':20.,'deadline_epoch':2200.,'deadline_monotonic':1220.,
                'maximum_child_rss_bytes':32_000_000_000,'owned_process_group_only':True,
                'gpu_leases_owned_by_parent':False,'watchdog_source':str(watchdog),
                'watchdog_source_sha256':source_manifest.file_sha256(watchdog),
                'parent_cmdline_sha256':hashlib.sha256(command).hexdigest()}
        _,permit=self.valid_config_permit();permit.update(parent=parent,parent_intent='intent.json',parent_intent_sha256='e'*64)
        with patch.object(paper_probe.os,'getppid',return_value=123),patch.object(paper_probe,'process_identity',return_value=parent),patch.object(paper_probe,'sealed_metadata',return_value=intent):
            with patch.object(Path,'read_bytes',return_value=command):paper_probe.verify_parent(permit)
            with patch.object(Path,'read_bytes',return_value=b'python\0unrelated.py\0'):
                with self.assertRaises(RuntimeError):paper_probe.verify_parent(permit)
            watchdog.write_text('# changed parent\n',encoding='utf-8')
            with self.assertRaises(ValueError):paper_probe.verify_parent(permit)

    def test_all_public_text_reads_writes_are_explicit_utf8(self):
        src=Path(__file__).parents[1]/'src';missing=[]
        for module in public_surface.MODULES:
            path=public_surface.module_file(src,module)
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'read_text','write_text'}:
                    encodings=[k.value for k in node.keywords if k.arg=='encoding']
                    if len(encodings)!=1 or not isinstance(encodings[0],ast.Constant) or encodings[0].value not in {'utf8','utf-8'}:
                        missing.append((module,node.lineno))
        self.assertEqual(missing,[])

    def test_large_same_size_header_preserved_weight_corruption_stops_before_load_or_lease(self):
        checkpoint=self.root/'checkpoint';checkpoint.mkdir()
        path=checkpoint/'model.safetensors'
        header=struct.pack('<Q',2)+b'{}'
        original=header+b'\0'*(2**20+32)
        path.write_bytes(original)
        receipt=self.root/'receipt.json'
        receipt.write_text(json.dumps({'files':[{'path':'model.safetensors','size':len(original),
            'sha256':hashlib.sha256(original).hexdigest(),'verified':True}]}),encoding='utf-8')
        config={'model':'pi0','factory':{'checkpoint':str(checkpoint),'asset_receipt':str(receipt)}}
        self.assertEqual(paper_probe.verify_probe_weights(config),1)
        changed=bytearray(original);changed[-1]=1;path.write_bytes(changed)
        self.assertEqual(path.stat().st_size,len(original));self.assertEqual(path.read_bytes()[:len(header)],header)
        state={'model_load_attempts':0,'model_loads':0}
        with patch.object(paper_probe,'cpu_factory',side_effect=AssertionError('native load must not occur')) as load,patch.object(paper_probe,'gpu_lease',side_effect=AssertionError('UUID lease must not occur')) as lease:
            with self.assertRaisesRegex(ValueError,'weight bytes differ'):
                paper_probe.materialize_cpu_once(config,state,lambda:None)
            load.assert_not_called();lease.assert_not_called()
        self.assertEqual((state['model_load_attempts'],state['model_loads']),(0,0))
        self.assertNotIn('torch',sys.modules)

    def test_complete_weight_verification_precedes_cpu_factory_and_uuid_lease_in_entry(self):
        path=Path(paper_probe.__file__)
        tree=ast.parse(path.read_text(encoding='utf-8'))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        locations={name:min(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Name) and n.func.id==name) for name in ('materialize_cpu_once','gpu_lease')}
        self.assertLess(locations['materialize_cpu_once'],locations['gpu_lease'])

    def test_qwen_vendor_hashes_must_match_qualified_cpu_and_sealed_processor_reference(self):
        from deepvision_vla.diagnostics.adapters.qwen_h16 import CRITICAL_SOURCE_FILES
        source_hashes={name:hashlib.sha256(name.encode()).hexdigest() for name in CRITICAL_SOURCE_FILES}
        processor_hashes={'config.json':'a'*64,'tokenizer.json':'b'*64}
        reference=self.root/'reference_source_manifest.json'
        reference.write_text(json.dumps({'files':[{'path':name,'sha256':value} for name,value in processor_hashes.items()]}),encoding='utf-8')
        config={'factory':{'source_sha256':source_hashes,'processor_sha256':processor_hashes,'config_dir':str(self.root)},
                'factory_metadata_seals':[{'path':str(reference),'sha256':source_manifest.file_sha256(reference)}]}
        cpu={'source_hashes':source_hashes}
        self.assertTrue(paper_probe.validate_qwen_vendor_binding(config,cpu)['source_hashes_equal_qualified_CPU_receipt'])
        with patch.object(paper_probe,'cpu_factory',side_effect=AssertionError('No model load')),patch.object(paper_probe,'gpu_lease',side_effect=AssertionError('No lease')):
            bad=deepcopy(config);bad['factory']['source_sha256'][next(iter(source_hashes))]='c'*64
            with self.assertRaisesRegex(ValueError,'qualified CPU receipt'):paper_probe.validate_qwen_vendor_binding(bad,cpu)
            bad=deepcopy(config);bad['factory']['processor_sha256']['config.json']='c'*64
            with self.assertRaisesRegex(ValueError,'sealed reference'):paper_probe.validate_qwen_vendor_binding(bad,cpu)
            reference.write_text('{"files":[]}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'SHA256'):paper_probe.validate_qwen_vendor_binding(config,cpu)
        self.assertNotIn('torch',sys.modules)


if __name__ == '__main__':
    unittest.main()
