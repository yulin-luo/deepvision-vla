"""Read-only formal reports. Default mode never opens model-result artifacts."""
from collections import Counter
from contextlib import ExitStack,contextmanager
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .formal_plan import PRIMARY, REPORT_METRICS as METRICS, TASK_PRIMARY, TASK_ROI_SCHEMA, TASK_ROI_SCOPE
from .formal_statistics import build_paired_panel, summarize_paired_panel, summarize_point_differences

MODELS=('openvla','pi0','qwen_h16')
LAYERS={'openvla':(0,10,20,30),'pi0':(0,6,12,16),'qwen_h16':(0,12,24,34)}
KEEPS=tuple(i/10 for i in range(11))
RANDOM=('roi_matched_random','random_nonROI_visual')
SEMANTIC=('roi_matched','non_target_object','empty_background')
SEMANTIC_PAIRS = (("roi_matched", "empty_background"),
                  ("roi_matched", "non_target_object"),
                  ("non_target_object", "empty_background"))
SUCCESS={'new_complete','history_reuse','within_run_alias'}
HORIZON={'openvla':1,'pi0':4,'qwen_h16':16}
LABELS={'official_bridge_orig_gt_mse':'Official Bridge-orig H1×7 GT MSE',
    'source_bound_H4_reference_mse':'Pi0 source-bound H4×7 dataset-reference MSE (not physical GT)',
    'native_H16_prediction_drift':'Qwen native H16×7 prediction drift MSE (not GT)',
    'normalized_H16_prediction_drift':'Qwen normalized H16×7 prediction drift MSE (not GT)'}
ANALYSIS_PLAN_SHA='cf5b73c0cb3a909a419bd9b4ea418412fc405585fc4c71c059830436b2dba88a'


class ReportIntegrityError(ValueError):pass


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def is_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def error_value(value):return float(value) if finite(value) and value>=0 else None


def read_sealed(descriptor,*,opener=None,json_value=True,maximum_bytes=None):
    if not isinstance(descriptor,dict) or not is_sha(descriptor.get('sha256')) or not descriptor.get('path'):
        raise ReportIntegrityError('Explicit artifact path and byte SHA required')
    if opener is not None:data=opener(descriptor['path'])
    elif maximum_bytes is not None:
        with Path(descriptor['path']).open('rb') as stream:data=stream.read(maximum_bytes+1)
    else:data=Path(descriptor['path']).read_bytes()
    if maximum_bytes is not None and len(data)>maximum_bytes:raise ReportIntegrityError('Scalar metadata exceeds its byte cap')
    if hashlib.sha256(data).hexdigest()!=descriptor['sha256']:raise ReportIntegrityError('Artifact seal differs: '+descriptor['path'])
    return json.loads(data.decode('utf-8')) if json_value else data


@contextmanager
def read_sealed_lines(descriptor):
    """Stream a sealed journal and verify its whole byte digest before reporting."""
    if not isinstance(descriptor,dict) or not is_sha(descriptor.get('sha256')) or not descriptor.get('path'):
        raise ReportIntegrityError('Explicit artifact path and byte SHA required')
    digest=hashlib.sha256()
    with Path(descriptor['path']).open('rb') as stream:
        def verified_lines():
            for line in stream:
                digest.update(line)
                yield line

        lines=verified_lines()
        yield lines
        # A corrupt journal stops the parser at its valid prefix. The artifact
        # still has to match the seal over the entire file, including the tail.
        for _ in lines:
            pass
        if digest.hexdigest()!=descriptor['sha256']:
            raise ReportIntegrityError('Artifact seal differs: '+descriptor['path'])


def validate_config(config):
    if (config.get('schema')!='formal1500_reporting_config_v1' or config.get('source_count')!=1500
            or set(config.get('models',{}))!=set(MODELS) or config.get('analysis_seed')!=20261114
            or config.get('bootstrap_resamples')!=2000 or config.get('physical_GT_promotion') is not False
            or config.get('analysis_plan',{}).get('sha256')!=ANALYSIS_PLAN_SHA):
        raise ReportIntegrityError('Fixed three-model/1500/preregistered analysis config required')
    return config


def admit_three_releases(config,*,opener=None):
    """Read only scalar parent metadata until every original job is released.

    Even hashing a prediction/request/reference file is deferred until this
    function succeeds. A gate failure contains metadata only and no curves.
    """
    validate_config(config);parents={};failures=[]
    for model in MODELS:
        spec=config['models'][model]
        try:
            parent=read_sealed(spec['parent_state'],opener=opener,maximum_bytes=8_000_000)
            if not isinstance(parent,dict):raise ReportIntegrityError('Parent terminal metadata must be an object')
            parents[model]=parent
            probe=parent.get('child_terminal_probe',{});child=probe.get('value',{})
            checks={'parent_not_complete_released':parent.get('status')=='complete_requested_inventory_independently_released',
                'independent_release_not_verified':parent.get('independent_release',{}).get('owned_resource_release_verified') is True,
                'child_inventory_not_qualified':parent.get('child_complete_inventory_qualified') is True,
                'child_probe_not_observed':probe.get('status')=='observed',
                'child_probe_SHA_differs':probe.get('sha256')==spec['child_result']['sha256'],
                'child_model_differs':child.get('model')==model,
                'child_not_complete':child.get('status')=='complete_requested_inventory',
                'child_scope_differs':bool(spec.get('scope_id')) and child.get('scope_id')==spec.get('scope_id'),
                'child_permit_differs':bool(spec.get('permit_id')) and child.get('permit_id')==spec.get('permit_id'),
                'child_run_bindings_differ':bool(spec.get('run_bindings')) and all(is_sha(v) for v in spec['run_bindings'].values()) and child.get('bindings')==spec['run_bindings']}
            reasons=[name for name,passed in checks.items() if not passed]
            if reasons:failures.append({'model':model,'reasons':reasons,'parent_status':parent.get('status'),
                'child_status':child.get('status'),'scalar_child_counts':child.get('counts'),
                'native_counters':child.get('actual_native_counters')})
        except (OSError,ValueError,TypeError,KeyError,AttributeError) as error:
            failures.append({'model':model,'reasons':['parent_metadata_unavailable_or_invalid'],'error':str(error)})
    if failures:return {'status':'metadata_gate_failure_no_values_read','admitted':False,'models':failures,'output_values_read':0},None
    # Only after all3 parent/release gates pass, read the scalar child result
    # bytes and verify equality to its hash-bound observed probe.
    children={}
    for model in MODELS:
        child=read_sealed(config['models'][model]['child_result'],opener=opener,maximum_bytes=8_000_000)
        if child!=parents[model]['child_terminal_probe']['value']:raise ReportIntegrityError('Child result differs from parent probe')
        children[model]=child
    return {'status':'all_three_original_jobs_complete_and_independently_released','admitted':True,
        'parent_state_SHA':{m:config['models'][m]['parent_state']['sha256'] for m in MODELS}},children


def chained_rows(data):
    """Independent chain reconciliation; preserve a corrupt tail as incomplete."""
    previous=None;sequence=0;offset=0
    lines=data.splitlines(keepends=True) if isinstance(data,bytes) else data
    for raw in lines:
        try:
            row=json.loads(raw.decode('utf-8'))
            if not isinstance(row,dict):raise ValueError('Journal row is not an object')
            unsigned={k:v for k,v in row.items() if k!='audit_sha256'}
            if (not raw.endswith(b'\n') or row.get('sequence')!=sequence or row.get('previous_sha256')!=previous
                    or canonical(unsigned)!=row.get('audit_sha256')):raise ValueError('Invalid sequence/hash/newline')
        except (ValueError,TypeError) as error:
            yield None,{'status':'partial_corrupt_tail','first_invalid_offset':offset,'valid_prefix_rows':sequence,'error':str(error)};return
        yield row,None;sequence+=1;previous=row['audit_sha256'];offset+=len(raw)


def plain_rows(data):
    lines=data.splitlines() if isinstance(data,bytes) else data
    for raw in lines:
        if not raw.strip():raise ReportIntegrityError('Blank request row cannot be ignored')
        value=json.loads(raw.decode('utf-8'))
        if not isinstance(value,dict):raise ReportIntegrityError('Request row is not an object')
        yield value


def reconcile_terminals(journals,requests):
    terminals={};issues=[];known=set(requests)
    for name,data in journals:
        for row,issue in chained_rows(data):
            if issue:issues.append({'file':name,**issue});continue
            if row.get('event')!='request_terminal':raise ReportIntegrityError('Non-terminal row in result journal')
            key=row.get('request_id')
            if key not in known or key in terminals:raise ReportIntegrityError('Foreign or duplicate terminal request')
            plan=requests[key]
            if any(row.get(k)!=plan.get(k) for k in ('ordinal','sample_id','model','region','layer','keep_ratio')):
                raise ReportIntegrityError('Terminal/request identity differs')
            terminals[key]=row
    return terminals,issues


def frozen_definitions(model,*,random=True,semantic=False,primary_region=PRIMARY):
    defs=[('baseline',None,None)]
    regions=[primary_region]+(list(SEMANTIC) if semantic else [])+(list(RANDOM) if random else [])
    return defs+[(region,layer,keep) for region in regions for layer in LAYERS[model] for keep in KEEPS]


def request_index(rows,inventory,model,*,random=True,semantic=False,primary_region=PRIMARY):
    """Only the frozen writer's original order/IDs and exact token counts."""
    definitions=frozen_definitions(model,random=random,semantic=semantic,primary_region=primary_region);result={};count=0
    for row in rows:
        if row.get('model') not in MODELS:raise ReportIntegrityError('Foreign model in original request stream')
        if row.get('model')!=model:continue
        ordinal=count//len(definitions)+1;index=count%len(definitions)
        if ordinal>len(inventory):raise ReportIntegrityError('Extra requested source')
        source=inventory[ordinal-1];region,layer,keep=definitions[index]
        key=f"{ordinal:04d}:{source['sample_id']}:{model}:{index}"
        if (row.get('request_id')!=key or row.get('ordinal')!=ordinal or row.get('sample_id')!=source['sample_id']
                or row.get('source_row_sha256')!=canonical(source) or row.get('image_sha256')!=source.get('image_sha256')
                or (row.get('region'),row.get('layer'),row.get('keep_ratio'))!=(region,layer,keep)):
            raise ReportIntegrityError('Frozen request order/layer/ratio differs')
        if primary_region == TASK_PRIMARY and row.get('roi_scope') != TASK_ROI_SCOPE:
            raise ReportIntegrityError('Task-ROI request scope differs from its manifest')
        if row.get('status') not in ('blocked','eligible_proxy'):raise ReportIntegrityError('Unknown planned status')
        if row['status']=='blocked':
            if not row.get('blocked_reasons'):raise ReportIntegrityError('Blocked request lacks reasons')
        elif region!='baseline':
            pool=row.get('pool_token_count');drop=row.get('distinct_dropped_tokens');retained=row.get('retained_token_count')
            if (type(pool) is not int or pool<=0 or type(drop) is not int or type(retained) is not int
                    or drop<0 or retained<0 or drop+retained!=pool
                    or row.get('actual_retained_fraction')!=retained/pool):raise ReportIntegrityError('Actual token budget differs')
            positions=row.get('positions')
            if (not isinstance(positions,list) or positions!=sorted(set(positions)) or len(positions)!=drop
                    or any(type(i) is not int or i<0 for i in positions)):raise ReportIntegrityError('Actual distinct mask/drop count differs')
        result[key]=row;count+=1
    if count!=len(inventory)*len(definitions):raise ReportIntegrityError('Incomplete original request denominator')
    # Each paired cohort shares its token budget, and every layer shares masks.
    budgets={};masks={}
    for row in result.values():
        if row['status']!='eligible_proxy' or row['region']=='baseline':continue
        value=(row['pool_token_count'],row['distinct_dropped_tokens'],row['retained_token_count'])
        group='paired_random' if row['region'] in RANDOM else 'semantic' if row['region'] in SEMANTIC else row['region']
        key=(row['sample_id'],group,row['keep_ratio'])
        if key in budgets and budgets[key]!=value:raise ReportIntegrityError('Cross-layer/paired-region token budgets differ')
        budgets[key]=value
        key=(row['sample_id'],row['region'],row['keep_ratio']);positions=tuple(row['positions'])
        if key in masks and masks[key]!=positions:raise ReportIntegrityError('Cross-layer masks differ')
        masks[key]=positions
    for sid in (s['sample_id'] for s in inventory):
        for region in [primary_region]+(list(RANDOM) if random else [])+(list(SEMANTIC) if semantic else []):
            previous=None
            for keep in KEEPS:
                value=masks.get((sid,region,keep))
                if value is not None:
                    if previous is not None and not set(value)<=set(previous):raise ReportIntegrityError('Frozen keep masks are not nested')
                    previous=value
    return result


def finite_chunk(value,horizon):
    return (isinstance(value,list) and len(value)==horizon and all(isinstance(row,list) and len(row)==7
        and all(finite(x) for x in row) for row in value))


def registered_metric(terminal,model,metric,reference_sha=None):
    if terminal is None:return None,'missing_terminal'
    if terminal.get('status') not in SUCCESS:return None,'terminal_not_verified_success:'+str(terminal.get('status'))
    prediction=terminal.get('prediction');metrics=terminal.get('metrics')
    prediction=prediction if isinstance(prediction,dict) else {};metrics=metrics if isinstance(metrics,dict) else {}
    space='normalized' if metric=='normalized_H16_prediction_drift' else 'native'
    if not finite_chunk(prediction.get(space),HORIZON[model]):return None,'missing_or_invalid_full_native_horizon'
    if model=='openvla':
        if metrics.get('gt_mapping_verified') is not True or metrics.get('gt_mse_status')!='verified_official_reference':
            return None,'official_GT_mapping_unavailable'
        value=metrics.get('gt_mse')
    elif model=='pi0':
        if (not reference_sha or metrics.get('reference_artifact_sha256')!=reference_sha
                or metrics.get('dataset_reference_status')!='verified_source_bound_dataset_reference'
                or metrics.get('global_physical_gt_verified') is not False):return None,'source_bound_H4_reference_unavailable_or_unbound'
        value=metrics.get('source_bound_dataset_reference_mse')
    else:
        if metrics.get('gt_mapping_verified') is True:raise ReportIntegrityError('Qwen global GT was unexpectedly promoted')
        field='normalized_prediction_drift' if space=='normalized' else 'native_prediction_drift'
        drift=metrics.get(field);value=drift.get('full_chunk_mse') if isinstance(drift,dict) else None
    number=error_value(value)
    return (number,None) if number is not None else (None,'missing_or_invalid_registered_metric')


def points(model,regions):return [f'{region}|L{layer}|K{index}' for region in regions for layer in LAYERS[model] for index in range(11)]
def point(row):return f"{row['region']}|L{row['layer']}|K{round(row['keep_ratio']*10)}"



def registered_layer_contrast(model, metric, group, regions, records, source_ids):
    first, last = LAYERS[model][0], LAYERS[model][-1]
    retention = [(region, k) for region in regions for k in range(len(KEEPS))]
    pairs = [(f"{region}|L{last}|K{k}", f"{region}|L{first}|K{k}") for region, k in retention]
    return {
        "model": model, "group": group, "metric": metric,
        "points": [f"{region}|K{k}" for region, k in retention],
        "comparison": "last_selected_minus_first_selected",
        "first_layer": first, "last_layer": last, "paired_sources": len(source_ids),
        "statistics": summarize_point_differences(records, source_ids, pairs),
    }


def registered_semantic_contrasts(model, metric, records, source_ids):
    coordinates = [(layer, k) for layer in LAYERS[model] for k in range(len(KEEPS))]
    contrasts = []
    for left, right in SEMANTIC_PAIRS:
        pairs = [(f"{left}|L{layer}|K{k}", f"{right}|L{layer}|K{k}") for layer, k in coordinates]
        contrasts.append({
            "model": model, "metric": metric, "left_region": left, "right_region": right,
            "points": [f"L{layer}|K{k}" for layer, k in coordinates],
            "paired_sources": len(source_ids),
            "statistics": summarize_point_differences(records, source_ids, pairs),
        })
    return contrasts


def build_model_report(inventory,requests,terminals,model,*,reference_sha=None,journal_issues=(),semantic_requested=False,
                        random_requested=True,primary_region=PRIMARY):
    ids=[s['sample_id'] for s in inventory];by_source={sid:[] for sid in ids}
    source_coverage=[];request_rows=[];counts=Counter();accounting=Counter();baseline={};group_regions={'primary':(primary_region,)}
    if random_requested:group_regions['paired_random']=RANDOM
    if semantic_requested:group_regions['semantic']=SEMANTIC
    for request_id,row in requests.items():
        sid=row['sample_id'];by_source[sid].append(row);terminal=terminals.get(request_id)
        status='missing_terminal' if terminal is None else terminal.get('status','unknown_status');counts[status]+=1
        entry=None if terminal is None else terminal.get('actual_attempted')
        accounting['actual_model_entries']+=entry is True;accounting['unknown_model_entries']+=entry is None and terminal is not None
        prediction=(terminal or {}).get('prediction');prediction=prediction if isinstance(prediction,dict) else {}
        accounting['finite_native_output_points']+=finite_chunk(prediction.get('native'),HORIZON[model])
        output={'model':model,'request_id':request_id,'ordinal':row['ordinal'],'sample_id':sid,
            **{k:row.get(k) for k in ('region','layer','keep_ratio','pool_token_count','distinct_dropped_tokens','retained_token_count','actual_retained_fraction')},
            'planned_status':row['status'],'blocked_reasons':row.get('blocked_reasons',[]),'terminal_status':status,
            'actual_model_entry':entry,'history_reuse':status=='history_reuse','within_run_alias':status=='within_run_alias',
            'computation_key':(terminal or {}).get('computation_key'),'metrics':{}}
        for metric in METRICS[model]:
            value,reason=registered_metric(terminal,model,metric,reference_sha)
            output['metrics'][metric]={'value':value,'missing_reason':reason}
        request_rows.append(output)
        if row['region']=='baseline':baseline[sid]=output
    out_by_id={row['request_id']:row for row in request_rows}
    panels={};per_source=[];shallow_deep=[];semantic_contrasts=[]
    for group,regions in group_regions.items():
        frozen=points(model,regions);qualified=[];records={}
        for sid in ids:
            selected=[r for r in by_source[sid] if r['region'] in regions]
            complete_plan=len(selected)==len(frozen) and all(r['status']=='eligible_proxy' for r in selected)
            if complete_plan:qualified.append(sid)
        for metric in METRICS[model]:
            records={}
            for sid in qualified:
                values={point(row):out_by_id[row['request_id']]['metrics'][metric]['value']
                    for row in by_source[sid] if row['region'] in regions}
                records[sid]={'baseline':baseline.get(sid,{}).get('metrics',{}).get(metric,{}).get('value'),'points':values}
            panel=build_paired_panel(records,ids,qualified,frozen);summary=summarize_paired_panel(panel)
            panels[group+'|'+metric]={'group':group,'metric':metric,'label':LABELS[metric],'regions':list(regions),**summary}
            for sid in ids:
                rec=records.get(sid)
                per_source.append({'model':model,'group':group,'metric':metric,'sample_id':sid,
                    'qualified':sid in qualified,'paired':sid in panel['paired_source_ids'],
                    'baseline':rec.get('baseline') if rec else None,'points':rec.get('points') if rec else None,
                    'signed_baseline_differences':{p:(v-rec['baseline'] if error_value(v) is not None and error_value(rec['baseline']) is not None else None)
                        for p,v in rec['points'].items()} if rec else None,
                    'experimental_unit':'source_episode'})
            source_ids = panel['paired_source_ids']
            shallow_deep.append(registered_layer_contrast(model, metric, group, regions, records, source_ids))
            if group == 'semantic':
                semantic_contrasts.extend(registered_semantic_contrasts(model, metric, records, source_ids))
    out_by_id={row['request_id']:row for row in request_rows}
    for source in inventory:
        rows=[out_by_id[row['request_id']] for row in by_source[source['sample_id']]]
        source_coverage.append({'ordinal':source['ordinal'],'sample_id':source['sample_id'],'model':model,'requested':len(rows),
            'planned_blocked':sum(r['planned_status']=='blocked' for r in rows),'planned_eligible':sum(r['planned_status']=='eligible_proxy' for r in rows),
            'terminal_status_counts':dict(Counter(r['terminal_status'] for r in rows)),
            'actual_model_entries':sum(r['actual_model_entry'] is True for r in rows),
            'unknown_model_entries':sum(r['actual_model_entry'] is None and r['terminal_status']!='missing_terminal' for r in rows),
            'missing_terminals':sum(r['terminal_status']=='missing_terminal' for r in rows)})
    requested=len(requests);eligible=sum(r['status']=='eligible_proxy' for r in requests.values())
    complete=(not journal_issues and len(terminals)==requested and not accounting['unknown_model_entries']
        and all(t.get('status') in SUCCESS|{'planned_blocked'} for t in terminals.values())
        and all(p['all_qualified_sources_metric_complete'] for p in panels.values()))
    return {'model':model,'status':'complete_metric_and_request_coverage' if complete else 'incomplete_coverage',
        'requested_sources':len(ids),'requested_points':requested,'qualified_points':eligible,'planned_blocked_points':requested-eligible,
        'terminal_status_counts':dict(counts),'journal_issues':list(journal_issues),'panels':panels,'shallow_deep':shallow_deep,
        'semantic_contrasts':semantic_contrasts,
        'actual_model_entries':accounting['actual_model_entries'],'unknown_model_entries':accounting['unknown_model_entries'],
        'finite_native_output_points':accounting['finite_native_output_points'],
        'metric_reference_coverage_points':{metric:sum(row['metrics'][metric]['value'] is not None for row in request_rows) for metric in METRICS[model]},
        'semantic_control_status':'requested_only_where_prequalified' if semantic_requested else 'not_requested',
        'paired_random_control_status':'requested' if random_requested else 'not_requested',
        'source_coverage':source_coverage,'request_rows':request_rows,'per_source_metrics':per_source,
        'layers':list(LAYERS[model]),'nominal_keep_ratios':list(KEEPS),'metric_spaces_not_pooled':True,
        'global_physical_GT_claim':False,'aliases_layers_ratios_are_new_replicates':False,
        'qwen_layer0_disclosure':'single-layer intervention; later native DeepStack reinjection branches unchanged' if model=='qwen_h16' else None}


def validated_reference_target(reference_artifact,codec_sha,codec,source,model):
    """Use only an already accepted full-horizon source-bound numeric target."""
    if not reference_artifact:return None,'no_bound_reference_artifact'
    if (reference_artifact.get('schema')!='verified_source_bound_dataset_references_v1'
            or reference_artifact.get('global_physical_gt_claim') is not False
            or reference_artifact.get('model_outcomes_used_for_reference_selection') is not False):
        raise ReportIntegrityError('Reference artifact cannot use model outcomes or promote physical GT')
    rows=[r for r in reference_artifact.get('references',[]) if r.get('model')==model and r.get('sample_id')==source['sample_id']]
    if len(rows)!=1:return None,'missing_or_ambiguous_source_reference'
    ref=rows[0];proofs=reference_artifact.get('accepted_full_trajectory_numeric_proof_sha256',[])
    if (ref.get('status')!='verified_numeric_reference' or ref.get('arrays_sha256')!=source.get('arrays_sha256')
            or ref.get('anchor')!=source.get('anchor') or ref.get('model_native_output_codec_sha256')!=codec_sha
            or not is_sha(ref.get('full_trajectory_numeric_proof_sha256')) or ref['full_trajectory_numeric_proof_sha256'] not in proofs):
        return None,'reference_source_or_proof_or_codec_unbound'
    if any(not is_sha(ref.get(k)) or ref[k]!=codec.get(k) for k in ('action_definition_sha256','statistics_sha256')):
        return None,'reference_action_definition_or_statistics_unbound'
    if (not isinstance(ref.get('reference_dataset'),str) or not ref['reference_dataset']
            or not isinstance(ref.get('reference_revision'),str) or not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}',ref['reference_revision'])
            or type(ref.get('converted_episode_index')) is not int or ref['converted_episode_index']<0):
        return None,'reference_dataset_revision_episode_unbound'
    horizon=HORIZON[model];frames=ref.get('converted_frame_indices',[])
    if (len(frames)!=horizon or any(type(k) is not int or k<0 for k in frames)
            or frames!=list(range(frames[0],frames[0]+horizon)) or not finite_chunk(ref.get('native_reference_horizon'),horizon)):
        return None,'reference_full_horizon_unavailable_no_padding'
    return ref['native_reference_horizon'],None


def channel_labels(codec,path):
    if not codec or not isinstance(path,list):return None
    value=codec
    for key in path:
        if not isinstance(value,dict) or key not in value:return None
        value=value[key]
    return value if isinstance(value,list) and len(value)==7 and all(isinstance(x,str) and x for x in value) else None


def per_channel_rows(inventory,requests,terminals,model,*,reference_artifact=None,reference_sha=None,codec=None,codec_sha=None,codec_channel_path=None):
    sources={s['sample_id']:s for s in inventory};baselines={};targets={}
    labels=channel_labels(codec,codec_channel_path)
    for row in requests.values():
        if row['region']=='baseline':baselines[row['sample_id']]=terminals.get(row['request_id'])
    for request_id,row in requests.items():
        if row['status']!='eligible_proxy':continue
        terminal=terminals.get(request_id);prediction=(terminal or {}).get('prediction');metric_data=(terminal or {}).get('metrics')
        prediction=prediction if isinstance(prediction,dict) else {};metric_data=metric_data if isinstance(metric_data,dict) else {}
        for metric in METRICS[model]:
            space='normalized' if metric=='normalized_H16_prediction_drift' else 'native'
            target=None;reason=None
            registered,reason=registered_metric(terminal,model,metric,reference_sha)
            if registered is not None:
                if model=='openvla':
                    value=metric_data.get('reference_action')
                    if isinstance(value,list) and len(value)==7 and all(finite(x) for x in value):target=[value]
                    else:reason='official_reference_action_not_preserved'
                elif model=='pi0':
                    sid=row['sample_id']
                    if sid not in targets:targets[sid]=validated_reference_target(reference_artifact,codec_sha,codec or {},sources[sid],model)
                    target,reason=targets[sid]
                else:
                    base=baselines.get(row['sample_id']);bp=(base or {}).get('prediction');target=bp.get(space) if isinstance(bp,dict) else None
                    if not finite_chunk(target,HORIZON[model]):target=None;reason='full_baseline_prediction_unavailable'
            values=prediction.get(space)
            if target is not None and not finite_chunk(values,HORIZON[model]):target=None;reason='full_prediction_unavailable'
            for channel in range(7):
                value=sum((values[t][channel]-target[t][channel])**2 for t in range(HORIZON[model]))/HORIZON[model] if target is not None else None
                yield {'model':model,'request_id':request_id,'sample_id':row['sample_id'],'region':row['region'],'layer':row['layer'],
                    'keep_ratio':row['keep_ratio'],'metric':metric,'channel_index':channel,'channel_label':labels[channel] if labels else None,
                    'channel_labels_status':'bound_codec_labels' if labels else 'codec_channel_labels_unavailable_indices_only',
                    'codec_sha256':codec_sha,'native_horizon':HORIZON[model],'channel_mse':value,'missing_reason':reason,
                    'global_physical_GT_claim':False}


def cluster_sensitivity(panel,inventory,field):
    """Optional predeclared container sensitivity; does not replace source mean."""
    if field is None:return {'status':'unavailable_no_predeclared_container_field','effective_clusters':None}
    lookup={s['sample_id']:s for s in inventory};clusters={}
    for i,sid in enumerate(panel['source_ids']):
        value=lookup[sid]
        for key in field.split('.'):
            if not isinstance(value,dict) or key not in value:return {'status':'unavailable_missing_container_metadata','effective_clusters':None}
            value=value[key]
        if not isinstance(value,str) or not value:return {'status':'unavailable_ambiguous_container_metadata','effective_clusters':None}
        clusters.setdefault(value,[]).append(i)
    if len(clusters)<2:return {'status':'insufficient_distinct_source_containers','effective_clusters':len(clusters)}
    # This function needs the fixed included-source values from the caller; the
    # public summary alone deliberately does not store raw prediction arrays.
    return {'status':'container_index_ready','effective_clusters':len(clusters),'field':field,'cluster_indices':list(clusters.values())}


def cluster_bootstrap(matrix,cluster_indices):
    import numpy as np
    values=np.asarray(matrix,dtype=np.float64);groups=[np.asarray(g,dtype=np.int64) for g in cluster_indices]
    if values.ndim!=2 or len(groups)<2 or not np.isfinite(values).all():raise ReportIntegrityError('Finite declared cluster matrix required')
    if sorted(int(i) for g in groups for i in g)!=list(range(len(values))):raise ReportIntegrityError('Container partition differs from fixed source cohort')
    rng=np.random.default_rng(20261114);draws=np.empty((2000,values.shape[1]))
    for b in range(2000):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))]);draws[b]=values[indices].mean(axis=0)
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    return {'status':'container_resampling_sensitivity_only','effective_clusters':len(groups),'resamples':2000,'analysis_seed':20261114,
        'lower':lo.tolist(),'upper':hi.tolist(),'interval_scope':'pointwise_not_simultaneous','does_not_replace_episode_weighted_primary':True}


def _write_csv(path,rows,fields):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(row.get(key),separators=(',',':')) if isinstance(row.get(key),(dict,list)) else row.get(key) for key in fields})


def write_report(output,models,inventory,*,admission,input_seals,per_channel=(),make_plots=True):
    out=Path(output);out.mkdir(exist_ok=False)
    report={'schema':'formal1500_report_v1','status':'complete_report' if all(m['status']=='complete_metric_and_request_coverage' for m in models.values()) else 'incomplete_coverage',
        'requested_source_identities':len(inventory),'admission':admission,'input_seals':input_seals,'models':{},
        'analysis_seed':20261114,'bootstrap_resamples':2000,'experimental_unit':'source_episode',
        'interval_scope':'pointwise95%_not_simultaneous','physical_GT_promotion':False,'model_calls':0,'GPU_calls':0,
        'cohorts_conditional_on_fixed_filtered_Bridge_panel':True,'negative_differences_preserved':True,'missing_replaced_with_zero':False}
    report['interpretation_limits']=['Incremental single-layer pathway intervention; previously transferred visual information can remain.',
        'AI-reviewed visible rectangles are relevance proxies, not physical-contact or global-state identity proofs.',
        'No closed-loop success, planning benefit or compositional generalization is inferred from offline curves.',
        'Action spaces and raw MSE magnitudes are not pooled across models.']
    request_rows=[];coverage=[];curves=[];shallow=[]
    with (out/'source_coverage.jsonl').open('x',encoding='utf-8') as sf,(out/'per_source_metrics.jsonl').open('x',encoding='utf-8') as pf:
        for model,result in models.items():
            for row in result['source_coverage']:sf.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            for row in result['per_source_metrics']:pf.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            request_rows.extend(result['request_rows'])
            coverage.append({'model':model,'status':result['status'],'requested_sources':result['requested_sources'],
                'requested_points':result['requested_points'],'qualified_points':result['qualified_points'],'planned_blocked_points':result['planned_blocked_points'],
                'actual_model_entries':result['actual_model_entries'],'unknown_model_entries':result['unknown_model_entries'],
                'finite_native_output_points':result['finite_native_output_points'],'metric_reference_coverage_points':result['metric_reference_coverage_points'],
                'terminal_status_counts':result['terminal_status_counts'],'native_counters':result.get('native_counters'),
                'semantic_control_status':result['semantic_control_status']})
            report['models'][model]={k:v for k,v in result.items() if k not in ('request_rows','source_coverage','per_source_metrics')}
            for panel in result['panels'].values():
                for i,p in enumerate(panel['points']):
                    for statistic in ('absolute_error','signed_baseline_difference'):
                        values=panel[statistic]
                        curves.append({'model':model,'group':panel['group'],'metric':panel['metric'],'label':panel['label'],'point':p,'statistic':statistic,
                            'mean':values['means'][i],'lower':values['lower'][i],'upper':values['upper'][i],
                            'requested_sources':panel['requested_sources'],'qualified_sources':panel['qualified_sources'],'paired_sources':panel['paired_sources'],
                            'coverage_status':panel['status'],'interval_scope':'pointwise95%_not_simultaneous'})
            for comparison in result['shallow_deep']:
                for i,p in enumerate(comparison['points']):
                    stats=comparison['statistics'];shallow.append({k:v for k,v in comparison.items() if k not in ('points','statistics')}
                        | {'point':p,'mean':stats['means'][i],'lower':stats['lower'][i],'upper':stats['upper'][i]})
    _write_csv(out/'coverage.csv',coverage,['model','status','requested_sources','requested_points','qualified_points','planned_blocked_points','actual_model_entries','unknown_model_entries','finite_native_output_points','metric_reference_coverage_points','terminal_status_counts','native_counters','semantic_control_status'])
    _write_csv(out/'requests.csv',request_rows,['model','request_id','ordinal','sample_id','region','layer','keep_ratio','pool_token_count','distinct_dropped_tokens','retained_token_count','actual_retained_fraction',
        'planned_status','blocked_reasons','terminal_status','actual_model_entry','history_reuse','within_run_alias','computation_key','metrics'])
    _write_csv(out/'curves.csv',curves,['model','group','metric','label','point','statistic','mean','lower','upper','requested_sources','qualified_sources','paired_sources','coverage_status','interval_scope'])
    _write_csv(out/'shallow_deep.csv',shallow,['model','group','metric','comparison','first_layer','last_layer','paired_sources','point','mean','lower','upper'])
    _write_csv(out/'per_channel.csv',per_channel,['model','request_id','sample_id','region','layer','keep_ratio','metric','channel_index','channel_label','channel_labels_status','codec_sha256','native_horizon','channel_mse','missing_reason','global_physical_GT_claim'])
    if make_plots:
        from .formal_reporting_plots import plot_report
        plot_report(report,out/'figures')
    with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    manifest={'schema':'formal1500_report_artifact_manifest_v1','model_calls':0,'GPU_calls':0,'input_seals':input_seals,
        'files':[{'path':str(p.relative_to(out)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]}
    with (out/'ARTIFACT_MANIFEST.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    return report


def attach_cluster_sensitivity(result,inventory,field):
    for panel in result['panels'].values():
        indices=cluster_sensitivity(panel,inventory,field)
        if indices['status']!='container_index_ready':panel['container_sensitivity']=indices;continue
        group,metric=panel['group'],panel['metric']
        records={r['sample_id']:r for r in result['per_source_metrics'] if r['group']==group and r['metric']==metric}
        values=[[records[sid]['points'][p] for p in panel['points']] for sid in panel['source_ids']]
        differences=[[value-records[sid]['baseline'] for value in row] for sid,row in zip(panel['source_ids'],values)]
        panel['container_sensitivity']={'field':field,'effective_clusters':indices['effective_clusters'],
            'absolute_error':cluster_bootstrap(values,indices['cluster_indices']),
            'signed_baseline_difference':cluster_bootstrap(differences,indices['cluster_indices'])}


def validate_analysis_permit(config,config_sha,permit):
    from . import formal_statistics
    expected_sources={'formal_reporting.py':file_sha(__file__),
        'formal_reporting_plots.py':file_sha(Path(__file__).with_name('formal_reporting_plots.py')),
        'formal_statistics.py':file_sha(formal_statistics.__file__)}
    if (permit.get('schema')!='formal1500_reporting_analysis_permit_v1' or permit.get('analysis_authorized') is not True
            or permit.get('config_sha256')!=config_sha or permit.get('source_sha256')!=expected_sources
            or permit.get('analysis_plan_sha256')!=config['analysis_plan']['sha256']
            or permit.get('parent_state_sha256')!={m:config['models'][m]['parent_state']['sha256'] for m in MODELS}
            or permit.get('child_result_sha256')!={m:config['models'][m]['child_result']['sha256'] for m in MODELS}):
        raise ReportIntegrityError('Separate root analysis authorization and exact sources/three-job seals required')
    return expected_sources


@contextmanager
def finite_scope(resources):
    """Finite Linux CPU-only report process; no child/model/GPU/lease exists."""
    import os,resource,signal,time
    if not hasattr(os,'sched_getaffinity'):raise ReportIntegrityError('Bounded analysis requires a Linux CPU runtime')
    validate_resources(resources,now=time.monotonic(),allowed=os.sched_getaffinity(0))
    affinity=resources['cpu_affinity'];end=resources['original_end_monotonic']
    previous=os.sched_getaffinity(0);old_limit=resource.getrlimit(resource.RLIMIT_AS);old_handler=signal.getsignal(signal.SIGALRM)
    old_timer=signal.getitimer(signal.ITIMER_REAL)
    if any(value>0 for value in old_timer) or (hasattr(signal,'sigpending') and signal.SIGALRM in signal.sigpending()):
        raise ReportIntegrityError('A pre-existing real timer/pending alarm cannot be replaced or disarmed')
    environment={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    old_environment={key:os.environ.get(key) for key in environment}
    soft=bounded_soft_limit(old_limit,resources['max_RSS_bytes'])
    def expired(*_):raise TimeoutError('Original report deadline exhausted; partial output preserved, no retry')
    def restore_environment():
        for key,value in old_environment.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    callbacks=[]
    # Every restoration is registered before its corresponding attempted
    # mutation, and the protective finally already covers the first operation.
    try:
        callbacks.append(lambda:os.sched_setaffinity(0,previous));os.sched_setaffinity(0,set(affinity))
        callbacks.append(restore_environment);os.environ.update(environment)
        callbacks.append(lambda:resource.setrlimit(resource.RLIMIT_AS,old_limit))
        resource.setrlimit(resource.RLIMIT_AS,(soft,old_limit[1]))  # HARD is unchanged; an existing lower SOFT is preserved.
        callbacks.append(lambda:signal.signal(signal.SIGALRM,old_handler));signal.signal(signal.SIGALRM,expired)
        remaining=end-time.monotonic()
        if remaining<=0:raise TimeoutError('Original report deadline expired during context setup')
        callbacks.append(lambda:signal.setitimer(signal.ITIMER_REAL,0));signal.setitimer(signal.ITIMER_REAL,remaining)
        yield
    finally:
        errors=[]
        for restore in reversed(callbacks):
            try:restore()
            except BaseException as error:errors.append(type(error).__name__+': '+str(error))
        if errors:raise ReportIntegrityError('Report resource-context restoration errors: '+repr(errors))


def bounded_soft_limit(old_limit,declared):
    if type(declared) is not int or declared<=0 or len(old_limit)!=2:raise ReportIntegrityError('Explicit positive memory cap required')
    finite_limits=[value for value in old_limit if value!=-1]
    if any(type(value) is not int or value<0 for value in finite_limits):raise ReportIntegrityError('Invalid original memory limits')
    return min([declared]+finite_limits)


def validate_resources(resources,*,now,allowed):
    affinity=resources.get('cpu_affinity');start=resources.get('original_started_monotonic');end=resources.get('original_end_monotonic')
    if (not isinstance(affinity,list) or len(affinity)!=len(set(affinity)) or not 1<=len(affinity)<=4
            or any(type(i) is not int or i<0 for i in affinity) or not set(affinity)<=set(allowed)
            or not finite(start) or not finite(end) or not start<=now<end or not 0<end-start<=900
            or resources.get('max_RSS_bytes')!=8_000_000_000):
        raise ReportIntegrityError('Declared availableCPU<=4/original900s/8GB reporting budget required')
    return resources


def validate_metric_artifact_bindings(spec,child,model):
    """Tie optional artifacts to the admitted original child BEFORE parsing."""
    reference=spec.get('references');codec=spec.get('codec')
    reference_sha=reference.get('sha256') if isinstance(reference,dict) else None
    if reference is not None and not isinstance(reference,dict):raise ReportIntegrityError('Malformed reference descriptor')
    if reference_sha!=child.get('source_bound_metric_reference_sha256'):
        raise ReportIntegrityError('Metric-reference artifact differs from the original child permit')
    if codec is not None:
        expected=child.get('bindings',{}).get('output_codec_sha256')
        if not isinstance(codec,dict) or not is_sha(expected) or codec.get('sha256')!=expected:
            raise ReportIntegrityError('Configured codec differs from admitted child output_codec_sha256')
    elif (model=='pi0' and reference is not None) or spec.get('codec_channel_path') is not None:
        raise ReportIntegrityError('Source-reference evaluation/named channel labels require the admitted codec')
    if model=='qwen_h16' and reference is not None:raise ReportIntegrityError('Qwen references remain unqualified in this report protocol')


def run(config_path,*,analyze=False,permit_path=None,permit_sha=None):
    # Default preparation reads exactly this configuration, not the paths it
    # contains. In particular no request/result/reference values are opened.
    config_raw=Path(config_path).read_bytes();config=json.loads(config_raw.decode('utf-8'));validate_config(config)
    if not analyze:
        if permit_path is not None or permit_sha is not None:raise ReportIntegrityError('Permit needs explicit analyze mode')
        return {'status':'prepared_not_authorized','output_values_read':0,'models':list(MODELS),'source_count':1500}
    if permit_path is None or not is_sha(permit_sha):raise ReportIntegrityError('Root analysis permit required')
    permit=read_sealed({'path':str(permit_path),'sha256':permit_sha})
    source_seals=validate_analysis_permit(config,hashlib.sha256(config_raw).hexdigest(),permit)
    with finite_scope(permit.get('resources',{})):
        return _execute_admitted(config,source_seals)


def _execute_admitted(config,source_seals):
    admission,children=admit_three_releases(config)
    if not admission['admitted']:return admission
    for model in MODELS:validate_metric_artifact_bindings(config['models'][model],children[model],model)
    # All three original jobs have now passed the release gate. Only this
    # section is permitted to touch request/result/reference prediction values.
    read_sealed(config['analysis_plan'],json_value=False)
    inventory=read_sealed(config['inventory'])
    if (len(inventory)!=1500 or [s.get('ordinal') for s in inventory]!=list(range(1,1501))
            or len({s.get('sample_id') for s in inventory})!=1500):raise ReportIntegrityError('Original ordered1500 inventory required')
    prepared={};models={};seals={'reporting_sources':source_seals,'analysis_plan':config['analysis_plan']['sha256'],'inventory':config['inventory']['sha256'],'models':{}}
    from .formal_execution_core import validate_manifest
    manifests={model:read_sealed(config['models'][model]['plan_manifest']) for model in MODELS}
    for model,manifest in manifests.items():
        validate_manifest(manifest,model)
        if 'pilot_source_ordinals' in manifest:
            raise ReportIntegrityError('Runtime pilot artifacts cannot enter the 1500-source population report')
    plan_schemas={manifest['schema'] for manifest in manifests.values()}
    if len(plan_schemas) > 1 and TASK_ROI_SCHEMA in plan_schemas:
        raise ReportIntegrityError('A task-ROI report requires the same protocol for all models')
    for model in MODELS:
        spec=config['models'][model];child=children[model]
        manifest=manifests[model]
        primary_region = TASK_PRIMARY if manifest['schema'] == TASK_ROI_SCHEMA else PRIMARY
        if (manifest['requests_jsonl_sha256']!=spec['requests']['sha256']
                or child['bindings'].get('requests_jsonl_sha256')!=spec['requests']['sha256']
                or child['bindings'].get('plan_manifest_sha256')!=spec['plan_manifest']['sha256']
                or child['bindings'].get('inventory_sha256')!=config['inventory']['sha256']):
            raise ReportIntegrityError('Original child/plan/request/inventory binding differs')
        with read_sealed_lines(spec['requests']) as lines:
            reqs=request_index(plain_rows(lines),inventory,model,random=manifest.get('include_random_controls') is True,
                semantic=manifest.get('include_controls') is True,primary_region=primary_region)
        terminal_names=child.get('terminal_files');terminal_shas=child.get('terminal_file_sha256')
        if (not isinstance(terminal_names,list) or len(set(terminal_names))!=len(terminal_names)
                or [Path(d['path']).name for d in spec['terminal_journals']]!=terminal_names
                or {Path(d['path']).name:d['sha256'] for d in spec['terminal_journals']}!=terminal_shas):
            raise ReportIntegrityError('Result/recovery journal set differs from child terminal probe')
        with ExitStack() as journal_stack:
            journals=[(Path(d['path']).name,journal_stack.enter_context(read_sealed_lines(d))) for d in spec['terminal_journals']]
            terminals,issues=reconcile_terminals(journals,reqs)
        # The event chain is independently checked; it is not used to invent
        # missing prediction records or reconstruct unknown model attempts.
        first_event=None;last_event=None
        with read_sealed_lines(spec['events']) as lines:
            for row,issue in chained_rows(lines):
                if issue:issues.append({'file':'events.jsonl',**issue})
                else:
                    if first_event is None:first_event=row
                    last_event=row
        if last_event is None or last_event.get('event')!='run_terminal':
            issues.append({'file':'events.jsonl','status':'missing_run_terminal'})
        if (first_event is None or first_event.get('event')!='authorized_run_intent'
                or any(first_event.get(k)!=child.get(k) for k in ('scope_id','permit_id','bindings'))):
            issues.append({'file':'events.jsonl','status':'original_run_intent_binding_differs'})
        parsed_counts=dict(Counter(row.get('status') for row in terminals.values()))
        if (parsed_counts!=child.get('counts') or len(terminals)!=child.get('terminal_records')
                or child.get('planned',{}).get('requested')!=len(reqs)
                or child.get('planned',{}).get('eligible')!=sum(r['status']=='eligible_proxy' for r in reqs.values())):
            issues.append({'status':'child_scalar_vs_original_journal_coverage_differs'})
        reference=read_sealed(spec['references']) if spec.get('references') else None
        codec=read_sealed(spec['codec']) if spec.get('codec') else None
        reference_sha=spec['references']['sha256'] if reference else None
        if model=='pi0' and reference:
            if codec is None:raise ReportIntegrityError('Pi source-bound references require the original sealed output codec')
            qualified={source['sample_id'] for source in inventory if validated_reference_target(reference,spec['codec']['sha256'],codec,source,model)[0] is not None}
            # This is a reporting-derived qualification check. Original journal
            # fields are immutable and remain referenced by their byte seals.
            for terminal in terminals.values():
                if terminal.get('sample_id') not in qualified and isinstance(terminal.get('metrics'),dict):
                    terminal['metrics']={**terminal['metrics'],'dataset_reference_status':'sealed_source_reference_not_qualified_for_reporting'}
        result=build_model_report(inventory,reqs,terminals,model,reference_sha=reference_sha,journal_issues=issues,
            semantic_requested=manifest.get('include_controls') is True,random_requested=manifest.get('include_random_controls') is True,
            primary_region=primary_region)
        result['native_counters']=child.get('actual_native_counters')  # absent stays unknown
        result['scalar_child_accounting']={key:child.get(key) for key in ('new_native_computations','new_computation_dispatches','new_actual_model_attempts',
            'new_attempts_with_unknown_entry','history_reuse_records','within_run_alias_records','model_load_count')}
        attach_cluster_sensitivity(result,inventory,config.get('cluster_field'))
        models[model]=result;prepared[model]=(reqs,terminals,reference,codec)
        seals['models'][model]={'parent_state':spec['parent_state']['sha256'],'child_result':spec['child_result']['sha256'],
            'plan_manifest':spec['plan_manifest']['sha256'],'requests':spec['requests']['sha256'],'events':spec['events']['sha256'],
            'terminal_journals':{Path(d['path']).name:d['sha256'] for d in spec['terminal_journals']},
            'references':reference_sha,'codec':spec['codec']['sha256'] if codec else None}
    def channels():
        for model,(reqs,terminals,reference,codec) in prepared.items():
            spec=config['models'][model]
            yield from per_channel_rows(inventory,reqs,terminals,model,reference_artifact=reference,
                reference_sha=spec['references']['sha256'] if reference else None,codec=codec,
                codec_sha=spec['codec']['sha256'] if codec else None,codec_channel_path=spec.get('codec_channel_path'))
    report=write_report(config['output'],models,inventory,admission=admission,input_seals=seals,per_channel=channels(),make_plots=True)
    return {'status':report['status'],'output':config['output'],'requested_source_identities':1500,'model_calls':0,'GPU_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--analyze',action='store_true');parser.add_argument('--permit',type=Path);parser.add_argument('--permit-sha256')
    args=parser.parse_args();result=run(args.config,analyze=args.analyze,permit_path=args.permit,permit_sha=args.permit_sha256)
    print(json.dumps(result,ensure_ascii=True))
    if args.analyze and result['status']!='complete_report':raise SystemExit(2 if result['status']=='metadata_gate_failure_no_values_read' else 1)


if __name__=='__main__':main()
