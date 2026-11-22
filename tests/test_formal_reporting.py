import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepvision_vla.diagnostics import formal_reporting as f


def digest(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()


def inventory(n=2):
    return [{'ordinal':i+1,'sample_id':f'synthetic-{i}', 'instruction':'d\u00e9plier le tissu de gauche \u00e0 droite',
        'image_sha256':digest('image'+str(i)),'arrays_sha256':digest('arrays'+str(i)),'anchor':2,
        'original_source':{'shard':'synthetic-container-'+str(i%2)}} for i in range(n)]


def fixture(model,n=2,blocked=(),*,random=True,semantic=False,primary_region=f.PRIMARY):
    sources=inventory(n);requests=[];terminals={};reference_sha=digest('synthetic-reference')
    for source in sources:
        sid=source['sample_id'];base=0. if model=='qwen_h16' else 2.+source['ordinal']
        for i,(region,layer,keep) in enumerate(f.frozen_definitions(
                model,random=random,semantic=semantic,primary_region=primary_region)):
            blocked_here=source['ordinal'] in blocked
            drop=None if region=='baseline' else 10-round(10*keep)
            positions=[] if region=='baseline' else list(range(100 if region==f.RANDOM[1] else 0,(100 if region==f.RANDOM[1] else 0)+drop))
            row={'request_id':f"{source['ordinal']:04d}:{sid}:{model}:{i}",'ordinal':source['ordinal'],'sample_id':sid,
                'source_row_sha256':f.canonical(source),'image_sha256':source['image_sha256'],'model':model,'region':region,'layer':layer,'keep_ratio':keep,
                'status':'blocked' if blocked_here else 'eligible_proxy','blocked_reasons':['synthetic_prequalified_block'] if blocked_here else [],
                'positions':None if blocked_here else positions,'pool_token_count':None if region=='baseline' else 10,
                'distinct_dropped_tokens':drop,'retained_token_count':None if drop is None else 10-drop,
                'actual_retained_fraction':None if drop is None else (10-drop)/10}
            if primary_region == f.TASK_PRIMARY:
                row['roi_scope'] = f.TASK_ROI_SCOPE
            requests.append(row)
            if blocked_here:
                terminals[row['request_id']]={**row,'status':'planned_blocked','actual_attempted':False};continue
            depth=0 if layer is None else f.LAYERS[model].index(layer)
            value=base if region=='baseline' else ((.5-.1*depth)*(1-keep) if model=='qwen_h16' else base-(.5+.1*depth)*(1-keep))
            if region in f.SEMANTIC:
                value += {'roi_matched':.3,'non_target_object':.1,'empty_background':0}[region]*(1-keep)
            native=[[math.sqrt(value)]*7 for _ in range(f.HORIZON[model])]
            prediction={'native':native};metrics={'gt_mapping_verified':False}
            if model=='openvla':metrics.update(gt_mse=value,gt_mapping_verified=True,gt_mse_status='verified_official_reference',reference_action=[0.]*7)
            elif model=='pi0':metrics.update(source_bound_dataset_reference_mse=value,reference_artifact_sha256=reference_sha,
                dataset_reference_status='verified_source_bound_dataset_reference',global_physical_gt_verified=False)
            else:
                prediction['normalized']=[[math.sqrt(2*value)]*7 for _ in range(16)]
                metrics.update(native_prediction_drift={'full_chunk_mse':value},normalized_prediction_drift={'full_chunk_mse':2*value})
            status='within_run_alias' if keep==1 else 'new_complete'
            terminals[row['request_id']]={**{k:row[k] for k in ('request_id','ordinal','sample_id','model','region','layer','keep_ratio')},
                'event':'request_terminal','status':status,'actual_attempted':status=='new_complete','prediction':prediction,'metrics':metrics,
                'computation_key':digest(sid+model+str(i))}
    return sources,f.request_index(requests,sources,model,random=random,semantic=semantic,
        primary_region=primary_region),terminals,reference_sha


class TaskROIReportTests(unittest.TestCase):
    def test_task_roi_preserves_complete_semantic_pairs_and_signed_contrasts(self):
        sources,requests,terminals,_ = fixture("openvla",blocked=(2,),random=False,semantic=True,
            primary_region=f.TASK_PRIMARY)
        report = f.build_model_report(sources,requests,terminals,"openvla",random_requested=False,
            semantic_requested=True,primary_region=f.TASK_PRIMARY)
        primary = next(panel for panel in report["panels"].values() if panel["group"]=="primary")
        self.assertEqual(primary["regions"], [f.TASK_PRIMARY])
        self.assertEqual(len(report["semantic_contrasts"]), 3)
        roi_background = report["semantic_contrasts"][0]
        self.assertEqual(roi_background["paired_sources"], 1)
        self.assertAlmostEqual(roi_background["statistics"]["means"][0], .3)
        self.assertAlmostEqual(roi_background["statistics"]["means"][10], 0.)

        rows = [dict(row) for row in requests.values()]
        row = next(row for row in rows if row["region"]=="empty_background" and row["ordinal"]==1)
        row.update(pool_token_count=20,retained_token_count=10,actual_retained_fraction=.5)
        with self.assertRaisesRegex(f.ReportIntegrityError, "token budgets differ"):
            f.request_index(rows,sources,"openvla",random=False,semantic=True,primary_region=f.TASK_PRIMARY)


def chain(rows):
    previous=None;out=[]
    for i,value in enumerate(rows):
        row={'sequence':i,'previous_sha256':previous,**value};row['audit_sha256']=f.canonical(row);previous=row['audit_sha256']
        out.append(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode('utf-8')+b'\n')
    return b''.join(out)


class SealedJournalStreamTests(unittest.TestCase):
    def test_corrupt_prefix_stop_still_verifies_remaining_bytes(self):
        prefix=chain([{'event':'authorized_run_intent'}])
        content=prefix+b'corrupt\nsealed tail\n'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            descriptor={'path':str(path),'sha256':hashlib.sha256(content).hexdigest()}
            path.write_bytes(content)
            with f.read_sealed_lines(descriptor) as lines:
                results=list(f.chained_rows(lines))
            self.assertEqual(results[0][0]['event'],'authorized_run_intent')
            self.assertEqual(results[1][1]['first_invalid_offset'],len(prefix))

            path.write_bytes(prefix+b'corrupt\nmodified tail\n')
            with self.assertRaisesRegex(f.ReportIntegrityError,'Artifact seal differs'):
                with f.read_sealed_lines(descriptor) as lines:
                    list(f.chained_rows(lines))


class AdmissionTests(unittest.TestCase):
    def setup_packet(self):
        files={};config={'schema':'formal1500_reporting_config_v1','source_count':1500,'models':{},'analysis_seed':20261114,
            'bootstrap_resamples':2000,'physical_GT_promotion':False,'analysis_plan':{'path':'plan','sha256':f.ANALYSIS_PLAN_SHA}}
        for model in f.MODELS:
            child={'status':'complete_requested_inventory','model':model,'scope_id':'scope','permit_id':'permit',
                'bindings':{'requests_jsonl_sha256':digest('requests')},'counts':{'new_complete':1},'actual_native_counters':None}
            childbytes=json.dumps(child,ensure_ascii=False).encode('utf-8');childsha=hashlib.sha256(childbytes).hexdigest()
            parent={'status':'complete_requested_inventory_independently_released','independent_release':{'owned_resource_release_verified':True},
                'child_complete_inventory_qualified':True,'child_terminal_probe':{'status':'observed','sha256':childsha,'value':child}}
            parentbytes=json.dumps(parent).encode('utf-8')
            files[model+'/parent']=parentbytes;files[model+'/child']=childbytes
            config['models'][model]={'parent_state':{'path':model+'/parent','sha256':hashlib.sha256(parentbytes).hexdigest()},
                'child_result':{'path':model+'/child','sha256':childsha},'scope_id':'scope','permit_id':'permit','run_bindings':child['bindings']}
        return config,files

    def test_unreleased_third_model_never_opens_any_child_or_prediction_path(self):
        config,files=self.setup_packet();parent=json.loads(files['qwen_h16/parent']);parent['status']='running'
        files['qwen_h16/parent']=json.dumps(parent).encode();config['models']['qwen_h16']['parent_state']['sha256']=hashlib.sha256(files['qwen_h16/parent']).hexdigest()
        accessed=[]
        def guarded(path):
            accessed.append(path)
            self.assertTrue(path.endswith('/parent'),'An active job must block even scalar child file reads')
            return files[path]
        gate,children=f.admit_three_releases(config,opener=guarded)
        self.assertFalse(gate['admitted']);self.assertIsNone(children);self.assertEqual(len(accessed),3);self.assertEqual(gate['output_values_read'],0)

    def test_all_three_parent_first_then_bound_child_probe(self):
        config,files=self.setup_packet();accessed=[]
        def opener(path):accessed.append(path);return files[path]
        gate,children=f.admit_three_releases(config,opener=opener)
        self.assertTrue(gate['admitted']);self.assertEqual(set(children),set(f.MODELS))
        self.assertEqual(accessed[:3],[m+'/parent' for m in f.MODELS])
        self.assertIsNone(children['openvla']['actual_native_counters'])

    def test_sha_or_release_false_is_metadata_failure_not_curve(self):
        config,files=self.setup_packet();config['models']['pi0']['parent_state']['sha256']='0'*64
        result,_=f.admit_three_releases(config,opener=files.__getitem__)
        self.assertFalse(result['admitted']);self.assertNotIn('panels',result)

    def test_default_prepared_reads_only_config_with_utf8_no_other_artifacts(self):
        config,_=self.setup_packet();config['models']={m:None for m in f.MODELS};config['instruction']='d\u00e9plier\u00e0'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json';path.write_text(json.dumps(config,ensure_ascii=False),encoding='utf-8')
            with patch.object(f,'read_sealed',side_effect=AssertionError('No referenced artifact may be read in default mode')):
                result=f.run(path)
            self.assertEqual(result['status'],'prepared_not_authorized');self.assertEqual(result['output_values_read'],0)
            self.assertEqual([p.name for p in Path(tmp).iterdir()],['config.json'])

    def test_finite_report_budget_original_clock_and_no_affinity_fallback(self):
        resources={'cpu_affinity':[24,25,26,27],'original_started_monotonic':100.,'original_end_monotonic':1000.,'max_RSS_bytes':8_000_000_000}
        self.assertIs(f.validate_resources(resources,now=200.,allowed=set(range(32))),resources)
        for field,value in [('original_end_monotonic',1001.),('cpu_affinity',[0,0]),('cpu_affinity',[0,1,2,3,4]),('max_RSS_bytes',9_000_000_000)]:
            bad={**resources,field:value}
            with self.assertRaises(f.ReportIntegrityError):f.validate_resources(bad,now=200.,allowed=set(range(32)))
        with self.assertRaises(f.ReportIntegrityError):f.validate_resources(resources,now=1000.,allowed=set(range(32)))
        with self.assertRaises(f.ReportIntegrityError):f.validate_resources(resources,now=200.,allowed={24,25,26})

    def test_analysis_permission_binds_exact_config_three_jobs_and_reporting_sources(self):
        from deepvision_vla.diagnostics import formal_statistics
        config,_=self.setup_packet()
        permit={'schema':'formal1500_reporting_analysis_permit_v1','analysis_authorized':True,'config_sha256':digest('synthetic-config'),
            'analysis_plan_sha256':f.ANALYSIS_PLAN_SHA,'source_sha256':{'formal_reporting.py':f.file_sha(f.__file__),
                'formal_reporting_plots.py':f.file_sha(Path(f.__file__).with_name('formal_reporting_plots.py')),
                'formal_statistics.py':f.file_sha(formal_statistics.__file__)},
            'parent_state_sha256':{m:config['models'][m]['parent_state']['sha256'] for m in f.MODELS},
            'child_result_sha256':{m:config['models'][m]['child_result']['sha256'] for m in f.MODELS}}
        self.assertEqual(f.validate_analysis_permit(config,digest('synthetic-config'),permit),permit['source_sha256'])
        bad=copy.deepcopy(permit);bad['parent_state_sha256']['pi0']='0'*64
        with self.assertRaises(f.ReportIntegrityError):f.validate_analysis_permit(config,digest('synthetic-config'),bad)
        bad=copy.deepcopy(permit);bad['analysis_authorized']=False
        with self.assertRaises(f.ReportIntegrityError):f.validate_analysis_permit(config,digest('synthetic-config'),bad)

    def test_codec_binding_rejects_before_any_reference_or_result_value_parse(self):
        config,files=self.setup_packet();children={m:json.loads(files[m+'/child']) for m in f.MODELS}
        spec=config['models']['openvla'];spec['codec']={'path':'must-never-be-opened.json','sha256':digest('different-codec')}
        children['openvla']['bindings']['output_codec_sha256']=digest('admitted-codec')
        with patch.object(f,'admit_three_releases',return_value=({'admitted':True},children)),patch.object(f,'read_sealed',side_effect=AssertionError('Artifact was parsed before codec binding')):
            with self.assertRaisesRegex(f.ReportIntegrityError,'output_codec_sha256'):f._execute_admitted(config,{})
        spec['codec']['sha256']=children['openvla']['bindings']['output_codec_sha256']
        f.validate_metric_artifact_bindings(spec,children['openvla'],'openvla')


class ResourceContextContracts(unittest.TestCase):
    def fixture(self,*,timer=(0.,0.),fail_setrlimit=False):
        from contextlib import ExitStack
        stack=ExitStack();state={'affinity':{24,25,26,27,28},'limit':(512*1024*1024,-1),'timer':timer,'handler':'original',
            'affinity_calls':0,'limit_calls':0,'timer_calls':0}
        old={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        def set_affinity(_pid,value):state['affinity_calls']+=1;state['affinity']=set(value)
        def set_limit(_kind,value):
            state['limit_calls']+=1;state['limit']=value
            if fail_setrlimit and state['limit_calls']==1:raise OSError('synthetic setup failure after partial mutation')
        def handler(_sig,value):before=state['handler'];state['handler']=value;return before
        def set_timer(_kind,value):state['timer_calls']+=1;state['timer']=(value,0.);return (0.,0.)
        resource=SimpleNamespace(RLIMIT_AS=1,RLIM_INFINITY=-1,getrlimit=lambda _:state['limit'],setrlimit=set_limit)
        stack.enter_context(patch.dict(sys.modules,{'resource':resource}))
        for module,name,value in [(os,'sched_getaffinity',lambda _:set(state['affinity'])),(os,'sched_setaffinity',set_affinity),
                (signal,'SIGALRM',14),(signal,'ITIMER_REAL',0),(signal,'getsignal',lambda _:state['handler']),
                (signal,'signal',handler),(signal,'getitimer',lambda _:state['timer']),(signal,'setitimer',set_timer),
                (signal,'sigpending',lambda:set()),(time,'monotonic',lambda:200.)]:
            stack.enter_context(patch.object(module,name,value,create=True))
        resources={'cpu_affinity':[24,25,26,27],'original_started_monotonic':100.,'original_end_monotonic':1000.,'max_RSS_bytes':8_000_000_000}
        return stack,state,old,resources

    def test_prior_lower_soft_and_all_environment_affinity_limits_restore(self):
        stack,state,old,resources=self.fixture()
        with stack:
            with f.finite_scope(resources):
                self.assertEqual(state['limit'],(512*1024*1024,-1));self.assertEqual(state['affinity'],{24,25,26,27})
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(state['limit'],(512*1024*1024,-1));self.assertEqual(state['affinity'],{24,25,26,27,28})
            self.assertEqual(state['handler'],'original');self.assertEqual(state['timer'],(0.,0.))
            self.assertEqual({key:os.environ.get(key) for key in old},old)
        self.assertEqual(f.bounded_soft_limit((-1,-1),8_000_000_000),8_000_000_000)
        self.assertEqual(f.bounded_soft_limit((1_000_000_000,2_000_000_000),8_000_000_000),1_000_000_000)

    def test_preexisting_timer_rejected_before_any_mutation_and_not_disarmed(self):
        stack,state,old,resources=self.fixture(timer=(17.,2.))
        with stack:
            with self.assertRaisesRegex(f.ReportIntegrityError,'pre-existing'): 
                with f.finite_scope(resources):self.fail('Body must not enter')
            self.assertEqual(state['affinity_calls'],0);self.assertEqual(state['limit_calls'],0);self.assertEqual(state['timer_calls'],0)
            self.assertEqual(state['timer'],(17.,2.));self.assertEqual({key:os.environ.get(key) for key in old},old)

    def test_partial_setup_failure_restores_from_before_first_mutation(self):
        stack,state,old,resources=self.fixture(fail_setrlimit=True)
        with stack:
            with self.assertRaisesRegex(OSError,'synthetic setup failure'):
                with f.finite_scope(resources):self.fail('Setup failure cannot reach body')
            self.assertEqual(state['limit'],(512*1024*1024,-1));self.assertEqual(state['affinity'],{24,25,26,27,28})
            self.assertEqual(state['handler'],'original');self.assertEqual(state['timer_calls'],0)
            self.assertEqual({key:os.environ.get(key) for key in old},old)


class ReportingTests(unittest.TestCase):
    def test_random88_shared_cohort_negative_delta_and_signed_shallow_deep(self):
        sources,requests,terminals,ref=fixture('openvla')
        report=f.build_model_report(sources,requests,terminals,'openvla')
        panel=report['panels']['paired_random|official_bridge_orig_gt_mse']
        self.assertEqual(len(panel['points']),88);self.assertEqual(panel['paired_sources'],2)
        self.assertLess(panel['signed_baseline_difference']['means'][0],0)
        self.assertEqual(panel['absolute_error']['lower'][0],panel['absolute_error']['lower'][44])
        sd=next(p for p in report['shallow_deep'] if p['group']=='paired_random')
        self.assertLess(sd['statistics']['means'][0],0)
        self.assertEqual(sd['statistics']['analysis_seed'],20261114)
        self.assertEqual(sd['statistics']['resamples'],2000)
        self.assertEqual(report['semantic_control_status'],'not_requested')

    def test_missing_metric_removes_source_from_entire_random_panel_not_other_points(self):
        sources,requests,terminals,ref=fixture('pi0')
        key=next(k for k,r in requests.items() if r['sample_id']==sources[0]['sample_id'] and r['region']==f.RANDOM[1] and r['keep_ratio']==.5)
        terminals[key]['metrics']['source_bound_dataset_reference_mse']=None
        report=f.build_model_report(sources,requests,terminals,'pi0',reference_sha=ref)
        panel=report['panels']['paired_random|source_bound_H4_reference_mse']
        self.assertEqual(panel['paired_sources'],1);self.assertEqual(panel['status'],'incomplete_metric_coverage')
        self.assertEqual(len(panel['excluded_qualified_sources']),1)
        self.assertEqual(report['status'],'incomplete_coverage')
        self.assertEqual(report['panels']['primary|source_bound_H4_reference_mse']['paired_sources'],2)

    def test_unbound_pi_reference_cannot_become_gt_and_qwen_spaces_separate(self):
        sources,requests,terminals,ref=fixture('pi0')
        self.assertIsNone(f.registered_metric(next(iter(terminals.values())),'pi0',f.METRICS['pi0'][0],None)[0])
        report=f.build_model_report(sources,requests,terminals,'pi0')
        self.assertEqual(report['panels']['primary|source_bound_H4_reference_mse']['paired_sources'],0)
        sources,requests,terminals,ref=fixture('qwen_h16')
        report=f.build_model_report(sources,requests,terminals,'qwen_h16')
        self.assertEqual(len(report['panels']),4)
        self.assertIn('later native DeepStack',report['qwen_layer0_disclosure'])
        for t in terminals.values():
            if t.get('metrics'):t['metrics']['gt_mapping_verified']=True;break
        with self.assertRaises(f.ReportIntegrityError):f.build_model_report(sources,requests,terminals,'qwen_h16')

    def test_aliases_and_reuse_are_points_not_replicates(self):
        sources,requests,terminals,ref=fixture('openvla')
        key=next(k for k,r in requests.items() if r['keep_ratio']==.5)
        terminals[key].update(status='history_reuse',actual_attempted=False)
        report=f.build_model_report(sources,requests,terminals,'openvla')
        self.assertEqual(report['requested_points'],266);self.assertLess(report['actual_model_entries'],266)
        self.assertEqual(report['terminal_status_counts']['history_reuse'],1)
        self.assertTrue(all(p['paired_sources']==2 for p in report['panels'].values()))
        self.assertFalse(report['aliases_layers_ratios_are_new_replicates'])

    def test_fixed_masks_token_budgets_and_utf8_source_hash_cannot_change(self):
        sources,requests,_,_=fixture('openvla');rows=list(requests.values())
        bad=copy.deepcopy(rows);r=next(r for r in bad if r['region']==f.RANDOM[1] and r['keep_ratio']==.5)
        r.update(pool_token_count=20,retained_token_count=15,actual_retained_fraction=.75)
        with self.assertRaises(f.ReportIntegrityError):f.request_index(bad,sources,'openvla')
        translated=copy.deepcopy(sources);translated[0]['instruction']='unfold left to right'
        with self.assertRaises(f.ReportIntegrityError):f.request_index(rows,translated,'openvla')
        self.assertEqual(sources[0]['instruction'].encode('utf-8'),b'd\xc3\xa9plier le tissu de gauche \xc3\xa0 droite')

    def test_journal_hashchain_recovery_duplicate_and_partial_tail(self):
        _,requests,terminals,_=fixture('openvla',n=1);rows=list(terminals.values())
        journals=[('results.jsonl',chain(rows[:50])),('recovery_results.jsonl',chain(rows[50:]))]
        result,issues=f.reconcile_terminals(journals,requests)
        self.assertEqual(len(result),133);self.assertFalse(issues)
        partial=[('results.jsonl',chain(rows[:50])+b'{partial')]
        result,issues=f.reconcile_terminals(partial,requests)
        self.assertEqual(len(result),50);self.assertEqual(issues[0]['status'],'partial_corrupt_tail')
        with self.assertRaises(f.ReportIntegrityError):f.reconcile_terminals(journals+[('dup',chain(rows[:1]))],requests)

    def test_unknown_entry_missing_metrics_and_foreign_requests_remain_explicit(self):
        sources,requests,terminals,_=fixture('openvla',n=1)
        first=next(iter(terminals.values()));first['actual_attempted']=None;first['metrics']=None
        result=f.build_model_report(sources,requests,terminals,'openvla')
        self.assertEqual(result['unknown_model_entries'],1);self.assertEqual(result['status'],'incomplete_coverage')
        self.assertIsNone(result['request_rows'][0]['metrics'][f.METRICS['openvla'][0]]['value'])
        foreign={**next(iter(requests.values())),'model':'unrequested-model'}
        with self.assertRaises(f.ReportIntegrityError):f.request_index([foreign],sources,'openvla')
        with self.assertRaises(f.ReportIntegrityError):list(f.plain_rows(b'{"x":1}\n\n'))

    def test_requested1500_complete_inventory_remains_visible_without_predictions(self):
        sources=inventory(1500);defs=f.frozen_definitions('openvla');rows=[]
        for source in sources:
            for i,(region,layer,keep) in enumerate(defs):
                rows.append({'request_id':f"{source['ordinal']:04d}:{source['sample_id']}:openvla:{i}",'ordinal':source['ordinal'],
                    'sample_id':source['sample_id'],'source_row_sha256':f.canonical(source),'image_sha256':source['image_sha256'],
                    'model':'openvla','region':region,'layer':layer,'keep_ratio':keep,'status':'blocked','blocked_reasons':['synthetic_no_annotation']})
        indexed=f.request_index(rows,sources,'openvla');report=f.build_model_report(sources,indexed,{},'openvla')
        self.assertEqual(report['requested_sources'],1500);self.assertEqual(report['requested_points'],199500)
        self.assertEqual(len(report['source_coverage']),1500);self.assertEqual(report['planned_blocked_points'],199500)
        self.assertEqual(report['terminal_status_counts']['missing_terminal'],199500);self.assertEqual(report['actual_model_entries'],0)
        self.assertEqual(report['panels']['primary|official_bridge_orig_gt_mse']['absolute_error']['means'],[None]*44)

    def test_seven_channel_full_H4_supplement_requires_sealed_reference(self):
        sources,requests,terminals,refsha=fixture('pi0',n=1);proof=digest('proof');codecsha=digest('codec')
        codec={'action_definition_sha256':digest('definition'),'statistics_sha256':digest('statistics'),'native_channels':['c'+str(i) for i in range(7)]}
        reference={'schema':'verified_source_bound_dataset_references_v1','global_physical_gt_claim':False,'model_outcomes_used_for_reference_selection':False,
            'accepted_full_trajectory_numeric_proof_sha256':[proof],'references':[{'model':'pi0','sample_id':sources[0]['sample_id'],
                'status':'verified_numeric_reference','arrays_sha256':sources[0]['arrays_sha256'],'anchor':2,
                'model_native_output_codec_sha256':codecsha,'full_trajectory_numeric_proof_sha256':proof,
                'reference_dataset':'synthetic-test-only','reference_revision':'a'*40,'converted_episode_index':0,
                'action_definition_sha256':codec['action_definition_sha256'],'statistics_sha256':codec['statistics_sha256'],
                'converted_frame_indices':[1,2,3,4],'native_reference_horizon':[[0.]*7 for _ in range(4)]}]}
        values=list(f.per_channel_rows(sources,requests,terminals,'pi0',reference_artifact=reference,reference_sha=refsha,
            codec=codec,codec_sha=codecsha,codec_channel_path=['native_channels']))
        self.assertEqual(len(values),133*7);self.assertEqual(values[0]['channel_label'],'c0');self.assertEqual(values[0]['native_horizon'],4)
        self.assertIsNotNone(values[0]['channel_mse']);self.assertFalse(values[0]['global_physical_GT_claim'])
        reference['references'][0]['native_reference_horizon'].pop()
        values=list(f.per_channel_rows(sources,requests,terminals,'pi0',reference_artifact=reference,reference_sha=refsha,codec=codec,codec_sha=codecsha))
        self.assertTrue(all(v['channel_mse'] is None for v in values))

    def test_source_container_bootstrap_is_sensitivity_not_primary(self):
        result=f.cluster_bootstrap([[1.,1.],[2.,2.],[10.,10.]],[[0,1],[2]])
        self.assertEqual(result['effective_clusters'],2);self.assertEqual(result['lower'][0],result['lower'][1])
        self.assertTrue(result['does_not_replace_episode_weighted_primary'])
        with self.assertRaises(f.ReportIntegrityError):f.cluster_bootstrap([[1.],[2.]],[[0],[0]])


class ReleasedMetricVerificationTests(unittest.TestCase):
    def test_manifested_partial_ledger_recomputes_paired_curves_and_semantic_contrasts(self):
        from deepvision_vla.diagnostics import formal_reporting_verify as verify

        sources, requests, terminals, _ = fixture(
            "openvla", n=3, blocked=(3,), random=False, semantic=True, primary_region=f.TASK_PRIMARY)
        missing = next(key for key, row in requests.items()
                       if row["ordinal"] == 2 and row["region"] == f.TASK_PRIMARY)
        del terminals[missing]
        result = f.build_model_report(
            sources, requests, terminals, "openvla",
            random_requested=False, semantic_requested=True, primary_region=f.TASK_PRIMARY)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "saved"
            f.write_report(folder, {"openvla": result}, sources, admission={},
                           input_seals={"synthetic_fixture_only": True}, make_plots=False)
            report_path = folder / "report.json"
            with patch("builtins.print") as output:
                verify.main(["--report", str(report_path)])
            receipt = json.loads(output.call_args.args[0])
            self.assertTrue(receipt["synthetic_fixture_only"])
            self.assertEqual((receipt["panels"], receipt["episode_rows"]), (2, 6))
            self.assertEqual((receipt["layer_contrasts"], receipt["semantic_contrasts"]), (2, 3))

            report = json.loads(report_path.read_bytes())
            report["models"]["openvla"]["semantic_contrasts"][0]["statistics"]["means"][0] += 0.01
            data = json.dumps(report).encode("utf8")
            report_path.write_bytes(data)
            manifest_path = folder / "ARTIFACT_MANIFEST.json"
            manifest = json.loads(manifest_path.read_bytes())
            descriptor = next(file for file in manifest["files"] if file["path"] == report_path.name)
            descriptor.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            manifest_path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Saved paired contrasts disagree"):
                verify.main(["--report", str(report_path)])


if __name__=='__main__':unittest.main(verbosity=2)
