"""Pure protocol/durability/cache tests; synthetic metadata is not NN proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from deepvision_vla.diagnostics import formal_execution_core as c


class FormalExecutionCoreContracts(unittest.TestCase):
    def permit(self):
        bound={k:'a'*64 for k in c.BINDINGS}
        return {'schema':c.SCHEMA,'execution_authorized':True,'permit_id':'fixture-permit','scope_id':'fixture-scope',
            'model':'pi0','seed':42,'reviewed_bank_qualified':True,'history_coverage_review_complete':True,
            'single_writer_scope':True,'bindings':bound,'budget_origin_epoch':100.,'cpu_deadline_epoch':200.,
            'gpu_deadline_epoch':300.,'absolute_deadline_epoch':300.,'max_total_distinct_attempts':10,
            'noise_identity_sha256':'n'.encode().hex().ljust(64,'0'),'resources':{'physical_gpu':2,
                'exclusive_gpu':True,'gpu_uuid':'GPU-12345678-1234-1234-1234-123456789abc','gpu_pci':'0000:01:00.0',
                'cpu_threads':8,'allocator_bytes':22_000_000_000,'observed_gpu_bytes_cap':24_000_000_000,
                'free_gpu_bytes_min':8_000_000_000,'load_free_gpu_bytes_min':32_000_000_000,
                'cpu_rss_hwm_cap':32_000_000_000,'cpu_mem_available_min':24_000_000_000,'outer_watchdog_required':True}}

    def test_separate_root_permit_exact_bindings_deadlines_and_resource_contract(self):
        permit=self.permit();c.validate_permit(permit,permit['bindings'],now=150.)
        for field,value in (('execution_authorized',False),('reviewed_bank_qualified',False),('history_coverage_review_complete',False),('cpu_deadline_epoch',140.)):
            bad=copy.deepcopy(permit);bad[field]=value
            with self.assertRaises(ValueError):c.validate_permit(bad,permit['bindings'],now=150.)
        bad=copy.deepcopy(permit);bad['resources']['cpu_threads']=2
        with self.assertRaises(ValueError):c.validate_permit(bad,permit['bindings'],now=150.)
        changed=dict(permit['bindings']);changed['profiles_sha256']='b'*64
        with self.assertRaises(ValueError):c.validate_permit(permit,changed,now=150.)

    def identity(self):
        source={'image_sha256':'a'*64,'arrays_sha256':'b'*64,'anchor':3,'instruction':'original text'}
        profile={'identity':{'model':'pi0','checkpoint_revision':c.REVISIONS['pi0'],'checkpoint_sha256':'c'*64,
            'source_sha256':'d'*64,'processor_sha256':'e'*64,'action_mapping_sha256':'f'*64}}
        row={'layer':6,'positions':[1,4],'region':'display_only','keep_ratio':.5}
        return source,profile,row,c.computation_identity(source,profile,row,self.permit())

    def test_nn_identity_separates_metric_only_gt_map_but_binds_all_actual_inputs(self):
        source,profile,row,identity=self.identity();changed=copy.deepcopy(profile)
        changed['identity']['action_mapping_sha256']='0'*64
        self.assertEqual(identity,c.computation_identity(source,changed,row,self.permit()))
        for field in ('image_sha256','arrays_sha256','anchor','instruction'):
            bad=dict(source);bad[field]='changed'
            if field in ('anchor','arrays_sha256','image_sha256'):
                with self.assertRaises(ValueError):c.computation_identity(bad,profile,row,self.permit())
            else:self.assertNotEqual(identity,c.computation_identity(bad,profile,row,self.permit()))
        bad=dict(row,positions=[1,5]);self.assertNotEqual(identity,c.computation_identity(source,profile,bad,self.permit()))

    def test_durable_intent_survives_missing_outcome_and_is_never_retried(self):
        _,_,_,identity=self.identity();key=c.canonical_sha(identity)
        scope={'scope_id':'s','origin':1,'cpu_deadline':2,'gpu_deadline':3}
        with tempfile.TemporaryDirectory()as directory:
            registry=c.AttemptRegistry(directory,scope);path=registry.reserve(key,identity,{'fixture':True})
            self.assertEqual(json.loads(path.read_text())['actual_attempted'],None)
            restarted=c.AttemptRegistry(directory,scope)
            with self.assertRaises(c.PriorAttemptExists)as error:restarted.reserve(key,identity,{'fixture':True})
            self.assertEqual(error.exception.state,'unknown_attempt_intent')
            with self.assertRaises(ValueError):c.AttemptRegistry(directory,{**scope,'cpu_deadline':200})

    def test_failed_outcome_is_retained_and_cannot_become_retry_or_cache_success(self):
        _,_,_,identity=self.identity();key=c.canonical_sha(identity)
        with tempfile.TemporaryDirectory()as directory:
            registry=c.AttemptRegistry(directory,{'scope_id':'s'});path=registry.reserve(key,identity,{})
            registry.outcome(path,status='error',actual_attempted=True,error={'type':'fixture'})
            with self.assertRaises(c.PriorAttemptExists)as error:registry.reserve(key,identity,{})
            self.assertEqual(error.exception.state,'error')
            with self.assertRaises(RuntimeError):registry.outcome(path,status='complete',actual_attempted=True)

    def test_explicit_new_scope_preserves_old_deadline_record_and_unknown_key(self):
        _,_,_,identity=self.identity();key=c.canonical_sha(identity)
        with tempfile.TemporaryDirectory()as directory:
            old=c.AttemptRegistry(directory,{'scope_id':'original','deadline':10});old.reserve(key,identity,{})
            new=c.AttemptRegistry(directory,{'scope_id':'separately-approved-new','deadline':20})
            with self.assertRaises(c.PriorAttemptExists):new.reserve(key,identity,{})
            self.assertEqual(len(list((Path(directory)/'scopes').glob('*.json'))),2)

    def test_append_only_journal_and_success_index_use_offsets_and_no_row_rewrite(self):
        _,_,_,identity=self.identity();key=c.canonical_sha(identity);prediction={'native':[[float(i)for i in range(7)]for _ in range(4)]}
        with tempfile.TemporaryDirectory()as directory:
            index=c.NewSuccessIndex(Path(directory)/'successes.jsonl')
            index.add(key,identity,prediction,'request-1');before=index.journal.path.read_bytes()
            actual,origin=index.get(key,identity);self.assertEqual(actual,prediction);self.assertEqual(origin,'request-1')
            self.assertEqual(index.journal.path.read_bytes(),before)
            with self.assertRaises(RuntimeError):index.add(key,identity,prediction,'request-2')
            index.close()

    def cache_fixture(self):
        _,_,row,identity=self.identity();key=c.canonical_sha(identity)
        prediction={'native':[[float(i)for i in range(7)]for _ in range(4)]}
        rawrow={'status':'complete','actual_attempted':True,'request':{'layer':row['layer'],'positions':row['positions']},'result':prediction}
        artifact={'status':'complete','lease_released':True,'model':'pi0','revision':c.REVISIONS['pi0'],'seed':42,'rows':[rawrow]}
        verification={'status':'verified_success_cache_export','model_outcomes_used_for_selection':False,
            'verified_computation_keys':[key],'verified_nn_identity_bindings':{key:identity},
            'verified_input_bindings':{key:{k:identity[k]for k in ('image_sha256','arrays_sha256','anchor','instruction')}}}
        entry={'status':'verified_success','computation_identity':identity,'computation_key':key,
            'verification_receipt_sha256':'a'*64,'origin':{'artifact_sha256':'b'*64,'row_pointer':['rows',0],
            'row_sha256':c.canonical_sha(rawrow)},'prediction':prediction,'prediction_sha256':c.canonical_sha(prediction)}
        return entry,{'b'*64:artifact},{'a'*64:verification}

    def test_history_requires_actual_success_origin_exact_payload_and_audited_identity(self):
        entry,artifacts,receipts=self.cache_fixture()
        key,result=c.verify_history_entry(entry,{},artifacts,receipts)
        self.assertEqual(key,entry['computation_key']);self.assertEqual(result,entry['prediction'])
        for mutate in ('unknown','altered_prediction','unknown_input','unreleased','not_actual_attempt'):
            e,a,r=copy.deepcopy((entry,artifacts,receipts))
            if mutate=='unknown':e['status']='unknown'
            elif mutate=='altered_prediction':e['prediction']['native'][0][0]=99.
            elif mutate=='unknown_input':r['a'*64]['verified_input_bindings']={}
            elif mutate=='unreleased':a['b'*64]['lease_released']=False
            else:a['b'*64]['rows'][0]['actual_attempted']=False
            with self.assertRaises(ValueError):c.verify_history_entry(e,{},a,r)

    def test_audited_legacy_normalization_must_link_unchanged_real_output_values(self):
        entry,artifacts,receipts=self.cache_fixture();key=entry['computation_key']
        raw={'status':'complete','model':'HaomingSong/lerobot-pi0-bridge','revision':c.REVISIONS['pi0'],
            'legacy_output':entry['prediction']['native']}
        artifacts['c'*64]=raw;export=artifacts['b'*64];export['normalized_export_source_sha256']='c'*64
        export['rows'][0]['source_prediction_binding']={'artifact_sha256':'c'*64,
            'native':{'pointer':['legacy_output'],'wrap_single_step':False}}
        entry['origin']['row_sha256']=c.canonical_sha(export['rows'][0])
        receipts['a'*64]['normalized_export_source_links']={'b'*64:'c'*64}
        c.verify_history_entry(entry,{},artifacts,receipts)
        wrong=copy.deepcopy(artifacts);wrong['c'*64]['legacy_output'][0][0]=99.
        with self.assertRaises(ValueError):c.verify_history_entry(entry,{},wrong,receipts)

    def test_partial_run_success_requires_separate_independent_terminated_prefix_acceptance(self):
        entry,artifacts,receipts=self.cache_fixture();artifacts['b'*64]['status']='partial'
        with self.assertRaises(ValueError):c.verify_history_entry(entry,{},artifacts,receipts)
        receipts['a'*64]['independently_accepted_terminated_partial_run_artifacts']=['b'*64]
        c.verify_history_entry(entry,{},artifacts,receipts)
        artifacts['b'*64]['status']='failed'
        with self.assertRaises(ValueError):c.verify_history_entry(entry,{},artifacts,receipts)

    def test_full_native_chunks_are_never_padded_or_truncated(self):
        with self.assertRaises(ValueError):c.prediction_payload({'native':[[0.]*7]},'pi0')
        with self.assertRaises(ValueError):c.prediction_payload({'native':[[0.]*6]*4},'pi0')

    def stream_fixture(self):
        profile={'identity':{'model':'pi0'}}
        sources=[{'sample_id':'metadata-fixture-'+str(i),'image_sha256':'a'*64,'arrays_sha256':'b'*64,
            'anchor':1,'instruction':'metadata only'}for i in range(1500)]
        manifest={'schema':'formal1500_data_only_request_plan_v1','launch_allowed':False,
            'requested_source_identities':1500,'models':['pi0'],'paper_layers':{'pi0':list(c.LAYERS['pi0'])},
            'keep_ratios':list(c.KEEPS),'membership':'patch_center','per_source_box_area_limit':.6,'seed':42,
            'synthetic_source_count':0,'include_controls':False}
        def rows(limit=1500):
            definitions=[('baseline',None,None)]+[(c.PRIMARY,l,k)for l in c.LAYERS['pi0']for k in c.KEEPS]
            for ordinal,source in enumerate(sources[:limit],1):
                for index,(region,layer,keep)in enumerate(definitions):
                    yield {'request_id':f"{ordinal:04d}:{source['sample_id']}:pi0:{index}",'ordinal':ordinal,'sample_id':source['sample_id'],'source_row_sha256':c.canonical_sha(source),
                        'image_sha256':source['image_sha256'],'model_identity_sha256':c.canonical_sha(profile['identity']),
                        'seed':42,'model':'pi0','region':region,'layer':layer,'keep_ratio':keep,'status':'blocked',
                        'blocked_reasons':['synthetic metadata qualification fixture'],'positions':None,
                        'launch_allowed':False,'paper_reproduction':False,'human_verified':False,'physical_contact_claim':False}
        return profile,sources,manifest,rows

    def test_all1500_streaming_metadata_rows_have_exact67500_coverage(self):
        profile,sources,manifest,rows=self.stream_fixture()
        result=c.validate_request_rows(rows(),manifest,sources,profile)
        self.assertEqual(result,{'requested':67500,'sources':1500,'requests_per_source':45,'blocked':67500,'eligible':0})
        with self.assertRaises(ValueError):c.validate_request_rows(rows(1),manifest,sources,profile)

    def test_boolean_layer_and_duplicate_baseline_cannot_pass_stream_gate(self):
        profile,sources,manifest,rows=self.stream_fixture();first=list(rows(1))
        bad=copy.deepcopy(first);bad[1]['layer']=False
        with self.assertRaises(ValueError):c.validate_request_rows(iter(bad),manifest,sources,profile)
        bad=copy.deepcopy(first);bad[1]=copy.deepcopy(bad[0])
        with self.assertRaises(ValueError):c.validate_request_rows(iter(bad),manifest,sources,profile)

    def test_interrupted_terminal_tail_is_preserved_and_valid_prefix_not_double_counted(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'terminals.jsonl';journal=c.DurableJSONL(path)
            journal.append('request_terminal',request_id='first',status='new_complete',actual_attempted=True)
            journal.close()
            with path.open('ab')as stream:stream.write(b'{"partial":')
            before=path.read_bytes();seen=c.scan_terminal_prefix(path)
            self.assertEqual(seen['request_ids'],{'first'});self.assertEqual(seen['actual_attempts'],1)
            self.assertIsNotNone(seen['preserved_invalid_tail']);self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
