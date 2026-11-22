"""Pure validation of predeclared paired random visual-control metadata.

Random non-ROI cells are spatial controls, never certified empty background or
semantically irrelevant content. No processor, model or runtime dispatch occurs.
"""
from dataclasses import asdict
import hashlib
import json
import re

from .controls import matched_control_mask
from .geometry import project_roi_boxes
from .sampling import dropped_positions

RANDOM_CONTROLS=('roi_matched_random','random_nonROI_visual')
METADATA_REGIONS=('roi_matched','random_nonROI_visual')
EXCLUSION='positive_intersection_with_any_selected_manipulated_object_robot_or_goal_box'
RELEVANT={'manipulated_object','robot_arm','robot_gripper','goal_context_anchor'}
PRIMARY_ROLES=RELEVANT-{'goal_context_anchor'}

def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def is_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def validate_bindings(value,models):
    if (not isinstance(value,dict) or value.get('schema')!='formal1500_random_control_evidence_v1'
            or value.get('seed')!=42 or type(value.get('seed')) is not int
            or any(not is_sha(value.get(k)) for k in ('metadata_sha256','protocol_sha256','source_bank_sha256'))):
        raise ValueError('Explicit sealed random metadata/protocol/bank and seed42 are required')
    geometry=value.get('geometry_file_sha256_by_model',{})
    if not isinstance(geometry,dict) or any(not is_sha(geometry.get(m)) for m in models):
        raise ValueError('Actual per-model geometry file SHA bindings required')
    return value

def strict_spatial_pools(layout,roles):
    """Historical center ROI, strict positive-intersection exclusion, no padding."""
    project=lambda boxes,membership:set(project_roi_boxes(boxes,image_sha256=layout.image_sha256,
        layout=layout,membership=membership)) if boxes else set()
    primary=project([r['xyxy_px'] for r in roles if r['role'] in PRIMARY_ROLES],'patch_center')
    relevant=[r['xyxy_px'] for r in roles if r['role'] in RELEVANT]
    centers=project(relevant,'patch_center');intersects=project(relevant,'positive_intersection')
    rows,cols=layout.grid_shape;cw,ch=layout.canvas_size;u0,v0,u1,v1=layout.content_xyxy
    valid=set()
    for row in range(rows):
        for col in range(cols):
            left,top,right,bottom=col*cw/cols,row*ch/rows,(col+1)*cw/cols,(row+1)*ch/rows
            if left>=u0 and top>=v0 and right<=u1 and bottom<=v1:
                valid.add(layout.token_positions[row*cols+col])
    strict=valid-intersects;center_pool=valid-centers
    return {'primary_patch_center_positions':sorted(primary),'relevant_patch_center_positions':sorted(centers),
        'relevant_positive_intersection_positions':sorted(intersects),'valid_content_visual_universe':sorted(valid),
        'strict_pixel_disjoint_random_pool_positions':sorted(strict),
        'strict_extra_excluded_positions_vs_center':sorted(center_pool-strict),
        'center_complement_token_count':len(center_pool),'center_comparison_possible_K':min(len(primary),len(center_pool)),
        'strict_proposed_K':min(len(primary),len(strict))}

def expected_subpools_and_schedule(source_id,model,pools,seed=42):
    k=pools['strict_proposed_K']
    if k<=0:return {},[],{}
    scoped=source_id+':'+model
    selected={'roi_matched':list(matched_control_mask(pools['primary_patch_center_positions'],required_count=k,
        sample_id=scoped,region='roi_fixed_subpool_K'+str(k),seed=seed)['positions']),
        'random_nonROI_visual':list(matched_control_mask(pools['strict_pixel_disjoint_random_pool_positions'],required_count=k,
        sample_id=scoped,region='random_strict_nonROI_fixed_subpool',seed=seed)['positions'])}
    schedule=[];masks={}
    for request_region,metadata_region in zip(RANDOM_CONTROLS,METADATA_REGIONS):
        for index in range(11):
            keep=index/10
            mask=list(dropped_positions(selected[metadata_region],keep_ratio=keep,sample_id=scoped,layer=0,seed=seed))
            schedule.append({'region':metadata_region,'ratio_index':index,'keep_ratio':keep,'drop_count':len(mask),
                'within_same_source_model_mask_sha256':sha(mask),'layer_independent':True})
            masks[(request_region,index)]=mask
    return selected,schedule,masks

def prepare_random_control(metadata,*,source,annotation,roles,layout,geometry,profile,bindings,layers,primary_reasons):
    reasons=list(primary_reasons)
    proposed=metadata.get('strict_proposed_K') if isinstance(metadata,dict) else None
    empty={'eligible':False,'reasons':reasons,'k':None,'masks':{},'metadata_row_sha256':sha(metadata),
        'subpools':{},'strict_pool_sha256':None,
        'metadata_proposed_K':proposed if type(proposed) is int and proposed>=0 else None}
    if metadata is None:return {**empty,'reasons':sorted(set(reasons+['random_control_metadata_missing']))}
    expected={'ordinal':source.get('ordinal'),'sample_id':source['sample_id'],'image_sha256':source['image_sha256'],
        'source_inventory_row_sha256':sha(source),'model':profile['identity']['model'],
        'control_kind':'random_nonROI_visual','semantic_background_claim':False,'human_verified':False,
        'policy_results_used':False,'source_bank_sha256':bindings['source_bank_sha256'],
        'model_identity_sha256':sha(profile['identity']),'processor_sha256':profile['identity']['processor_sha256'],
        'actual_geometry_file_sha256':bindings['geometry_file_sha256_by_model'][profile['identity']['model']],
        'primary_membership':'patch_center','random_exclusion_rule':EXCLUSION,
        'logical_layers_same_schedule':list(layers),'NN_dispatch_authorized':False,
        'runtime_GT_and_probe_target_gates_not_checked':True,'new_processor_calls':0}
    if any(sha(metadata.get(key))!=sha(value) for key,value in expected.items()):
        reasons.append('random_control_source_bank_model_or_protocol_unbound')
    if not isinstance(metadata.get('blocked_reasons'),list) or any(not isinstance(r,str) or not r for r in metadata.get('blocked_reasons',[])):
        reasons.append('random_control_blocked_status_malformed')
    elif metadata.get('annotation_geometry_eligible_only') is not True:
        reasons+=metadata['blocked_reasons'] or ['random_control_annotation_geometry_not_qualified']
    elif metadata['blocked_reasons']:reasons.append('random_control_eligibility_status_conflicts')
    if layout is None or geometry is None:
        reasons.append('random_control_actual_geometry_missing')
    else:
        if (sha(metadata.get('actual_layout'))!=sha(asdict(layout))
                or metadata.get('actual_geometry_audit_sha256')!=geometry.get('audit_sha256')):
            reasons.append('random_control_actual_geometry_receipt_unbound')
    raw_roles=annotation.get('roles') if isinstance(annotation,dict) else None
    if (not isinstance(raw_roles,list) or metadata.get('relevant_roles_sha256')!=sha(raw_roles)
            or (annotation.get('roles_sha256',sha(raw_roles))!=sha(raw_roles))):
        reasons.append('random_control_original_role_list_unbound')
    if reasons:return {**empty,'reasons':sorted(set(reasons))}
    pools=strict_spatial_pools(layout,roles)
    if any(sha(metadata.get(key))!=sha(value) for key,value in pools.items()):
        reasons.append('random_control_strict_pixel_or_primary_pool_mismatch')
    k=pools['strict_proposed_K']
    if k<=0:reasons.append('random_control_matched_budget_empty')
    selected,schedule,masks=expected_subpools_and_schedule(source['sample_id'],profile['identity']['model'],pools)
    if (sha(metadata.get('selected_fixed_subpools'))!=sha(selected) or sha(metadata.get('keep_ratio_mask_schedule'))!=sha(schedule)):
        reasons.append('random_control_fixed_seed_nested_schedule_unbound')
    return {'eligible':not reasons,'reasons':sorted(set(reasons)),'k':k,'masks':masks if not reasons else {},
        'metadata_row_sha256':sha(metadata),'subpools':selected if not reasons else {},
        'metadata_proposed_K':k,
        'strict_pool_sha256':sha(pools['strict_pixel_disjoint_random_pool_positions'])}

def validate_random_request(row,metadata,bindings):
    """Core stream check against exact predeclared masks; does not run geometry."""
    proof=row.get('random_control_provenance',{})
    if (proof.get('metadata_file_sha256')!=bindings['metadata_sha256']
            or proof.get('metadata_row_sha256')!=sha(metadata)
            or proof.get('protocol_sha256')!=bindings['protocol_sha256']
            or row.get('request_cohort')!='paired_random_visual'
            or row.get('semantic_background_claim') is not False):
        raise ValueError('Random request provenance/cohort is unbound')
    if row['status']=='blocked':return
    if metadata is None:raise ValueError('Eligible random request requires its sealed source/model metadata row')
    if (metadata.get('sample_id')!=row['sample_id'] or metadata.get('model')!=row['model']
            or metadata.get('source_inventory_row_sha256')!=row['source_row_sha256']
            or metadata.get('image_sha256')!=row['image_sha256']
            or metadata.get('model_identity_sha256')!=row['model_identity_sha256']
            or metadata.get('source_bank_sha256')!=bindings['source_bank_sha256']
            or metadata.get('actual_geometry_file_sha256')!=bindings['geometry_file_sha256_by_model'][row['model']]):
        raise ValueError('Random metadata identity does not bind the request')
    if proof.get('strict_pixel_disjoint_pool_sha256')!=sha(metadata.get('strict_pixel_disjoint_random_pool_positions')):
        raise ValueError('Strict random pool provenance differs')
    if metadata.get('annotation_geometry_eligible_only') is not True or metadata.get('blocked_reasons'):
        raise ValueError('Random request cannot promote blocked metadata')
    k=metadata.get('strict_proposed_K');region=row['region'];meta_region=METADATA_REGIONS[RANDOM_CONTROLS.index(region)]
    subpool=metadata.get('selected_fixed_subpools',{}).get(meta_region)
    if type(k) is not int or k<=0 or not isinstance(subpool,list) or len(subpool)!=k:
        raise ValueError('Random paired K/subpool is missing')
    index=round(row['keep_ratio']*10);mask=list(dropped_positions(subpool,keep_ratio=row['keep_ratio'],
        sample_id=row['sample_id']+':'+row['model'],layer=0,seed=42))
    expected_schedule={'region':meta_region,'ratio_index':index,'keep_ratio':row['keep_ratio'],
        'drop_count':len(mask),'within_same_source_model_mask_sha256':sha(mask),'layer_independent':True}
    if (expected_schedule not in metadata.get('keep_ratio_mask_schedule',[]) or row['positions']!=mask
            or row.get('pool_token_count')!=k or row.get('distinct_dropped_tokens')!=len(mask)
            or row.get('retained_token_count')!=k-len(mask)
            or row.get('actual_retained_fraction')!=(k-len(mask))/k):
        raise ValueError('Random mask differs from the fixed source/region/seed schedule')
