"""Pure permit, streaming-plan, durable-intent and verified-cache contracts.

No imports of Torch/model backends, no dispatch, no process signals. A data-only
plan never grants execution. Root-issued permits and prior verification receipts
are external authorization/evidence artifacts, all bound by their exact bytes.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .formal_plan import LAYERS,REVISIONS,KEEPS,PRIMARY,CONTROLS,canonical_sha,is_sha
from .formal_plan import PILOT_KEEPS, REPORT_METRICS, TASK_PRIMARY, TASK_ROI_BOX_ROLES, TASK_ROI_SCHEMA, TASK_ROI_SCOPE, validate_pilot_ordinals
from .adapters.qwen_h16_contracts import matches_controlled_repository
from .geometry import VisualLayout
from .random_controls import RANDOM_CONTROLS,validate_bindings,validate_random_request

SCHEMA='formal1500_single_model_execution_permit_v1'
BINDINGS=('config_sha256','plan_manifest_sha256','requests_jsonl_sha256','inventory_sha256',
    'profiles_sha256','annotations_sha256','geometries_sha256','scientific_source_sha256',
    'runtime_sha256','asset_receipt_sha256','output_codec_sha256','history_manifest_sha256')
UUID=re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
PCI=re.compile(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]')
MODEL_IDS={'openvla':('openvla','openvla/openvla-7b'),'pi0':('pi0','HaomingSong/lerobot-pi0-bridge'),
    'qwen_h16':('qwen_h16',)}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def sealed_json(path,expected):
    if not is_sha(expected)or file_sha(path)!=expected:raise ValueError('Exact artifact SHA differs: '+str(path))
    return json.loads(Path(path).read_text(encoding='utf8'))


def restore_visual_layout(value):
    fields=('source_size','crop_xyxy','canvas_size','content_xyxy','grid_shape','token_positions')
    if not isinstance(value,dict)or set(value)!={'image_sha256',*fields}:
        raise ValueError('Exact serialized VisualLayout fields required')
    restored={'image_sha256':value['image_sha256']}
    for field in fields:
        if not isinstance(value[field],(list,tuple)):raise ValueError('Serialized layout sequence is malformed: '+field)
        restored[field]=tuple(value[field])
    return VisualLayout(**restored)


def validate_permit(permit,observed,*,now=None):
    now=time.time()if now is None else now
    if (permit.get('schema')!=SCHEMA or permit.get('execution_authorized')is not True
            or permit.get('model')not in LAYERS or permit.get('seed')!=42
            or permit.get('reviewed_bank_qualified')is not True or permit.get('history_coverage_review_complete')is not True
            or permit.get('single_writer_scope')is not True or not permit.get('permit_id')or not permit.get('scope_id')):
        raise ValueError('A separate explicit reviewed root execution permit is required')
    bound=permit.get('bindings',{})
    if set(bound)!=set(BINDINGS)or any(not is_sha(bound[k])or observed.get(k)!=bound[k]for k in BINDINGS):
        raise ValueError('Permit does not bind exact plan/source/profile/config/history bytes')
    times=[permit.get(k)for k in ('budget_origin_epoch','cpu_deadline_epoch','gpu_deadline_epoch','absolute_deadline_epoch')]
    if any(type(x)not in(int,float)or not math.isfinite(x)for x in times):raise ValueError('Original absolute scope times are required')
    origin,cpu,gpu,end=times
    if not origin<=now<cpu<=end or not origin<gpu<=end or now>=gpu:
        raise ValueError('Original permit CPU/GPU budget has expired or was reset')
    resource=permit.get('resources',{})
    if (type(resource.get('physical_gpu'))is not int or not 0<=resource['physical_gpu']<=7
            or resource.get('exclusive_gpu')is not True or not UUID.fullmatch(str(resource.get('gpu_uuid','')))
            or not PCI.fullmatch(str(resource.get('gpu_pci','')))or resource.get('cpu_threads')!=8
            or resource.get('allocator_bytes')!=22_000_000_000 or resource.get('observed_gpu_bytes_cap')!=24_000_000_000
            or resource.get('free_gpu_bytes_min')!=8_000_000_000 or resource.get('load_free_gpu_bytes_min')!=32_000_000_000
            or resource.get('cpu_rss_hwm_cap')!=32_000_000_000 or resource.get('cpu_mem_available_min')!=24_000_000_000
            or resource.get('outer_watchdog_required')is not True):
        raise ValueError('Explicit exclusive physicalGPU/UUID/PCI and truthful8CPU resource bounds required')
    if type(permit.get('max_total_distinct_attempts'))is not int or permit['max_total_distinct_attempts']<=0:
        raise ValueError('Finite cumulative unique attempt cap is required')
    if not is_sha(permit.get('noise_identity_sha256')):raise ValueError('Explicit seed/noise identity is required even for deterministic models')
    if permit.get('metric_reference_sha256')is not None and not is_sha(permit['metric_reference_sha256']):
        raise ValueError('Optional dataset reference is a separate exact metric-only SHA binding')
    return permit


def validate_manifest(manifest,model):
    schema=manifest.get('schema')
    pilot_scope = 'pilot_source_ordinals' in manifest
    # Previously sealed pilots advertised the full grid. Both pilot grids stay distinct from a population run.
    valid_keeps = (list(KEEPS), list(PILOT_KEEPS)) if pilot_scope else (list(KEEPS),)
    if (schema not in {'formal1500_data_only_request_plan_v1','formal1500_data_only_request_plan_v2',TASK_ROI_SCHEMA}
            or manifest.get('launch_allowed')is not False
            or manifest.get('requested_source_identities')!=1500 or model not in manifest.get('models',[])
            or manifest.get('paper_layers',{}).get(model)!=list(LAYERS[model])
            or manifest.get('keep_ratios') not in valid_keeps or manifest.get('membership')!='patch_center'
            or manifest.get('per_source_box_area_limit')!=.6 or manifest.get('seed')!=42
            or manifest.get('synthetic_source_count')!=0):
        raise ValueError('The original 1500-source inventory and a registered four-layer data-only protocol are required')
    random=schema=='formal1500_data_only_request_plan_v2'
    if random:
        if (manifest.get('include_random_controls')is not True
                or type(manifest.get('include_controls'))is not bool
                or manifest.get('random_control_regions')!=list(RANDOM_CONTROLS)
                or manifest.get('optional_random_control_requests_per_source_model')!=88
                or manifest.get('random_control_semantic_background_claim')is not False
                or manifest.get('cohort_request_budgets')!={'primary':45,'semantic':132 if manifest.get('include_controls')else 0,'paired_random_visual':88}):
            raise ValueError('Explicit v2 paired random-control protocol required')
        validate_bindings(manifest.get('random_control_evidence'),manifest['models'])
    elif manifest.get('include_random_controls'):
        raise ValueError('Random controls require the v2 protocol')
    if schema == TASK_ROI_SCHEMA and (
            type(manifest.get('include_controls')) is not bool
            or manifest.get('primary_region') != TASK_PRIMARY
            or manifest.get('primary_role_union') != list(TASK_ROI_BOX_ROLES)):
        raise ValueError('The task-ROI protocol requires the explicit object, robot and interaction union')
    if 'report_metrics' in manifest:
        models = manifest['models']
        if (schema != TASK_ROI_SCHEMA or any(value not in REPORT_METRICS for value in models)
                or manifest['report_metrics'] != {value: list(REPORT_METRICS[value]) for value in models}):
            raise ValueError('Report metrics differ from the model action references')
    if pilot_scope:
        validate_pilot_ordinals(manifest['pilot_source_ordinals'])
        if (schema != TASK_ROI_SCHEMA or manifest.get('eligible_for_population_report') is not False
                or manifest.get('include_controls') is not False):
            raise ValueError('A runtime pilot is restricted to the primary task ROI')
    return 1+4*len(manifest['keep_ratios'])+(132 if manifest.get('include_controls')else 0)+(88 if random else 0)


def iter_selected_requests(path,model):
    with Path(path).open('rb')as stream:
        for line in stream:
            if not line.strip():raise ValueError('Blank plan rows are not silently skipped')
            row=json.loads(line)
            if row.get('model')==model:yield row


def validate_stream(path,manifest,inventory,profile,*,random_control_metadata_path=None):
    if file_sha(path)!=manifest['requests_jsonl_sha256']:raise ValueError('Plan stream bytes changed')
    metadata=None
    if manifest.get('include_random_controls'):
        if random_control_metadata_path is None:raise ValueError('v2 requires the exact random metadata file')
        if file_sha(random_control_metadata_path)!=manifest['random_control_evidence']['metadata_sha256']:
            raise ValueError('Random metadata bytes changed')
        raw=Path(random_control_metadata_path).read_text(encoding='utf8')
        metadata=json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(s) for s in raw.splitlines() if s.strip()]
    elif random_control_metadata_path is not None:raise ValueError('v1 has no random metadata input')
    return validate_request_rows(iter_selected_requests(path,profile['identity']['model']),manifest,inventory,profile,
        random_control_metadata=metadata)


def validate_request_rows(rows,manifest,inventory,profile,*,random_control_metadata=None):
    model=profile['identity']['model'];n=validate_manifest(manifest,model)
    if len(inventory)!=1500 or len({r['sample_id']for r in inventory})!=1500:
        raise ValueError('Exactly1500 original ordered source identities required')
    pilot = 'pilot_source_ordinals' in manifest
    source_ordinals = tuple(manifest['pilot_source_ordinals']) if pilot else range(1, 1501)
    current=source_ordinals[0];seen=set();count=0;blocked=0
    task_roi = manifest.get('schema') == TASK_ROI_SCHEMA
    primary_region = TASK_PRIMARY if task_roi else PRIMARY
    keeps = manifest['keep_ratios']
    definitions={('baseline',None,None)}|{(primary_region,l,k)for l in LAYERS[model]for k in keeps}
    if manifest.get('include_controls'):definitions|={(r,l,k)for r in CONTROLS for l in LAYERS[model]for k in KEEPS}
    ordered=[('baseline',None,None)]+[(primary_region,l,k)for l in LAYERS[model]for k in keeps]
    if manifest.get('include_controls'):ordered += [(r,l,k)for r in CONTROLS for l in LAYERS[model]for k in KEEPS]
    random=manifest.get('include_random_controls')is True;random_index={}
    if random:
        if not isinstance(random_control_metadata,list):raise ValueError('v2 requires sealed random source/model metadata rows')
        definitions|={(r,l,k)for r in RANDOM_CONTROLS for l in LAYERS[model]for k in KEEPS}
        ordered += [(r,l,k)for r in RANDOM_CONTROLS for l in LAYERS[model]for k in KEEPS]
        original_ids={s['sample_id'] for s in inventory}
        for value in random_control_metadata:
            ident=(value.get('sample_id'),value.get('model'))
            if ident in random_index or ident[0]not in original_ids or ident[1]not in LAYERS:
                raise ValueError('Duplicate/foreign random metadata cannot replace a source')
            random_index[ident]=value
    for row in rows:
        ordinal=row.get('ordinal')
        source_index = count // n
        if source_index >= len(source_ordinals) or ordinal != source_ordinals[source_index]:
            raise ValueError('Source request order/coverage changed')
        if ordinal!=current:
            if seen!=definitions:raise ValueError('Source request order/coverage changed')
            current=ordinal;seen=set()
        if not 1<=ordinal<=1500:raise ValueError('Foreign source ordinal')
        source=inventory[ordinal-1];key=(row.get('region'),row.get('layer'),row.get('keep_ratio'))
        if row.get('region')!='baseline'and(type(row.get('layer'))is not int or type(row.get('keep_ratio'))not in(int,float)):
            raise ValueError('Layer/keep fields must be explicit numeric values, not booleans')
        if key not in definitions or key in seen:raise ValueError('Missing/duplicate/foreign layer-ratio request')
        if key!=ordered[len(seen)]or row.get('request_id')!=f"{ordinal:04d}:{source['sample_id']}:{model}:{len(seen)}":
            raise ValueError('Actual request order/ID differs from the frozen writer protocol')
        if task_roi and row.get('roi_scope') != TASK_ROI_SCOPE:
            raise ValueError('Task-ROI request scope differs from its manifest')
        if 'report_metrics' in manifest and row.get('metric') != REPORT_METRICS[model][0]:
            raise ValueError('Primary report metric differs from its manifest')
        if not seen and key!=('baseline',None,None):raise ValueError('Every source baseline must precede its dependent requests')
        seen.add(key)
        if (row.get('sample_id')!=source['sample_id']or row.get('source_row_sha256')!=canonical_sha(source)
                or row.get('image_sha256')!=source['image_sha256']or row.get('model_identity_sha256')!=canonical_sha(profile['identity'])
                or row.get('seed')!=42 or row.get('launch_allowed')is not False
                or row.get('paper_reproduction')is not False or row.get('human_verified')is not False
                or row.get('physical_contact_claim')is not False):raise ValueError('Request/source/model identity differs from sealed plan')
        if row.get('gt_mse_allowed')is True and(model!='openvla'or profile.get('action_mapping',{}).get('gt_mapping_verified')is not True):
            raise ValueError('Plan cannot promote unverified global Pi/Qwen GT via a request flag')
        if random:
            expected_cohort='paired_random_visual' if row['region']in RANDOM_CONTROLS else 'primary' if row['region']in {'baseline',PRIMARY} else 'semantic'
            if row.get('request_cohort')!=expected_cohort:raise ValueError('Request control cohorts cannot be conflated')
            if row['region']in RANDOM_CONTROLS:
                validate_random_request(row,random_index.get((source['sample_id'],model)),manifest['random_control_evidence'])
        if row.get('status')=='blocked':
            if pilot:raise ValueError('Runtime pilot includes a blocked request')
            if row.get('positions')is not None or not row.get('blocked_reasons'):raise ValueError('Blocked requests need unknown masks and explicit reasons')
            blocked+=1
        elif row.get('status')=='eligible_proxy':
            positions=row.get('positions')
            if (not isinstance(positions,list)or positions!=sorted(set(positions))
                    or any(type(x)is not int or x<0 for x in positions)
                    or row.get('distinct_dropped_tokens')!=len(positions)):
                raise ValueError('Exact distinct mask positions are required')
        else:raise ValueError('Unknown planning status')
        count+=1
    if seen!=definitions or count!=len(source_ordinals)*n:raise ValueError('Partial plan cannot authorize its declared workload')
    return {'requested':count,'sources':len(source_ordinals),'requests_per_source':n,'blocked':blocked,'eligible':count-blocked}


def computation_identity(source,profile,row,permit):
    identity=profile['identity'];model=identity['model']
    required=('checkpoint_sha256','source_sha256','processor_sha256')
    if identity.get('checkpoint_revision')!=REVISIONS[model]or any(not is_sha(identity.get(k))for k in required):
        raise ValueError('Actual model/checkpoint/source/processor identity is unknown')
    if (not is_sha(source.get('image_sha256'))or not is_sha(source.get('arrays_sha256'))
            or type(source.get('anchor'))is not int or source['anchor']<0
            or not isinstance(source.get('instruction'),str)or not source['instruction'].strip()):
        raise ValueError('Exact source array/anchor/instruction required')
    positions=row.get('positions')
    if (not isinstance(positions,list)or positions!=sorted(set(positions))or any(type(x)is not int or x<0 for x in positions)
            or (row['layer']is not None and row['layer']not in LAYERS[model])):
        raise ValueError('Blocked/unknown/foreign masks have no executable computation identity')
    return {'schema':'formal_native_computation_identity_v1','image_sha256':source['image_sha256'],
        'arrays_sha256':source['arrays_sha256'],'anchor':source['anchor'],'instruction':source['instruction'],
        'model':model,'checkpoint_revision':identity['checkpoint_revision'],
        **{k:identity[k]for k in required},'asset_receipt_sha256':permit['bindings']['asset_receipt_sha256'],
        'runtime_sha256':permit['bindings']['runtime_sha256'],'output_codec_sha256':permit['bindings']['output_codec_sha256'],
        'seed':42,'noise_identity_sha256':permit['noise_identity_sha256'],'layer':row['layer'],'positions':list(positions)}


def prediction_payload(value,model):
    horizon={'openvla':1,'pi0':4,'qwen_h16':16}[model];result={}
    for field in (('native','normalized')if model=='qwen_h16'else('native',)):
        array=value.get(field)
        if (not isinstance(array,list)or len(array)!=horizon or any(not isinstance(r,list)or len(r)!=7 for r in array)
                or any(type(x)not in(int,float)or not math.isfinite(x)for r in array for x in r)):
            raise ValueError('Complete finite native action chunk required; no padding/truncation')
        result[field]=array
    return result


class DurableJSONL:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('xb');self.sequence=0;self.previous=None
        sync_directory(self.path.parent)
    def append(self,event,**data):
        row={'sequence':self.sequence,'previous_sha256':self.previous,'event':event,**data}
        row['audit_sha256']=canonical_sha(row);raw=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
        offset=self.stream.tell();self.stream.write(raw);self.stream.flush();os.fsync(self.stream.fileno())
        self.sequence+=1;self.previous=row['audit_sha256'];return offset,len(raw)
    def close(self):self.stream.close()


class PriorAttemptExists(RuntimeError):
    def __init__(self,state,path):super().__init__('Prior known/unknown attempt may not be retried');self.state=state;self.path=str(path)


class AttemptRegistry:
    """Per-key O_EXCLJSONL prevents duplicate dispatch without NFSflock locks."""
    def __init__(self,root,scope):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        if not isinstance(scope.get('scope_id'),str)or not scope['scope_id']:raise ValueError('Explicit scope identity required')
        scopes=self.root/'scopes';scopes.mkdir(exist_ok=True)
        metadata=scopes/(canonical_sha(scope['scope_id'])+'.json')
        raw=(json.dumps(scope,sort_keys=True,separators=(',',':'))+'\n').encode()
        try:
            with metadata.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            sync_directory(metadata.parent)
        except FileExistsError:
            if metadata.read_bytes()!=raw:raise ValueError('Scope/original deadlines changed across resumption')
        self.existing_count=sum(1 for _ in self.root.glob('[0-9a-f][0-9a-f]/*.jsonl'))
    def path(self,key):
        if not is_sha(key):raise ValueError('Computation SHA key required')
        return self.root/key[:2]/(key+'.jsonl')
    def reserve(self,key,identity,owner):
        path=self.path(key);path.parent.mkdir(exist_ok=True)
        try:stream=path.open('xb')
        except FileExistsError:
            raise PriorAttemptExists(self.inspect(key)['status'],path)
        intent={'event':'attempt_intent','key':key,'identity':identity,'owner':owner,'actual_attempted':None,'epoch':time.time()}
        stream.write((json.dumps(intent,separators=(',',':'))+'\n').encode());stream.flush();os.fsync(stream.fileno());stream.close()
        sync_directory(path.parent)
        self.existing_count+=1;return path
    def inspect(self,key):
        path=self.path(key)
        if not path.exists():return None
        try:
            rows=[json.loads(line)for line in path.read_text(encoding='utf-8').splitlines()if line.strip()]
            if not rows:return {'status':'unknown_empty_reservation','path':str(path)}
            if rows[0].get('event')!='attempt_intent'or rows[0].get('key')!=key or canonical_sha(rows[0].get('identity'))!=key:
                return {'status':'unknown_corrupt_intent','path':str(path)}
            if len(rows)==1:return {'status':'unknown_attempt_intent','path':str(path),'intent':rows[0]}
            if len(rows)!=2 or rows[1].get('event')!='attempt_outcome':return {'status':'unknown_corrupt_outcome','path':str(path)}
            return {'status':rows[1].get('status','unknown_outcome'),'path':str(path),'intent':rows[0],'outcome':rows[1]}
        except (OSError,ValueError,TypeError):return {'status':'unknown_unreadable_reservation','path':str(path)}
    def outcome(self,path,*,status,actual_attempted,prediction=None,error=None):
        rows=[json.loads(line)for line in Path(path).read_text(encoding='utf-8').splitlines()if line.strip()]
        if len(rows)!=1 or rows[0].get('event')!='attempt_intent':raise RuntimeError('Attempt outcome already exists/intent is corrupt')
        row={'event':'attempt_outcome','status':status,'actual_attempted':actual_attempted,'epoch':time.time(),
            'prediction':prediction,'error':error}
        with Path(path).open('ab')as stream:stream.write((json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())


def resolve_pointer(value,pointer):
    for key in pointer:
        if type(key)not in(str,int):raise ValueError('Only actual dict/list pointers are supported')
        value=value[key]
    return value


def artifact_model_matches(artifact, model):
    name = artifact.get('model', artifact.get('repo'))
    return name in MODEL_IDS[model] or (model == 'qwen_h16' and matches_controlled_repository(name))


def verify_history_entry(entry,manifest,artifacts,verification_receipts):
    """Verify exact raw successful result, source bits and root-audited NN identity."""
    identity=entry.get('computation_identity',{});key=canonical_sha(identity)
    if (identity.get('model')not in REVISIONS or identity.get('checkpoint_revision')!=REVISIONS[identity['model']]
            or identity.get('seed')!=42 or type(identity.get('anchor'))is not int or identity['anchor']<0
            or not isinstance(identity.get('instruction'),str)or not identity['instruction'].strip()):
        raise ValueError('Historical native model revision/seed/source inputs are unknown')
    if entry.get('status')!='verified_success'or entry.get('computation_key')!=key:
        raise ValueError('Unknown/failed/cache-key-mismatched history cannot be reused')
    if not all(is_sha(identity.get(k))for k in ('image_sha256','arrays_sha256','checkpoint_sha256','source_sha256',
            'processor_sha256','asset_receipt_sha256','runtime_sha256','output_codec_sha256','noise_identity_sha256')):
        raise ValueError('Historical NN input/model/asset/runtime/codec fields are unknown')
    receipt=verification_receipts.get(entry.get('verification_receipt_sha256'))
    if (not receipt or receipt.get('status')!='verified_success_cache_export'
            or receipt.get('model_outcomes_used_for_selection')is not False
            or key not in receipt.get('_verified_key_set',receipt.get('verified_computation_keys',[]))):
        raise ValueError('A sealed actual history verification receipt is required')
    if receipt.get('verified_nn_identity_bindings',{}).get(key)!=identity:
        raise ValueError('Historical asset/scientific/runtime/codec/noise identity was not explicitly audited')
    origin=entry.get('origin',{});artifact=artifacts.get(origin.get('artifact_sha256'))
    partial_accepted=origin.get('artifact_sha256')in receipt.get('independently_accepted_terminated_partial_run_artifacts',[])
    if (not artifact or artifact.get('lease_released')is not True
            or artifact.get('status')not in('complete','complete_requested_inventory','partial')
            or(artifact.get('status')=='partial'and not partial_accepted)):
        raise ValueError('Raw prior run must have completed and released its lease')
    row=resolve_pointer(artifact,origin['row_pointer'])
    if (row.get('status')!='complete'or row.get('actual_attempted')is not True
            or canonical_sha(row)!=origin.get('row_sha256')or artifact.get('model')!=identity.get('model')
            or artifact.get('revision')!=identity.get('checkpoint_revision')):
        raise ValueError('Raw historical row is not an actual successful same-model computation')
    request=row['request']
    if request.get('layer')!=identity.get('layer')or request.get('positions')!=identity.get('positions'):
        raise ValueError('History intervention differs')
    if request.get('seed',artifact.get('seed'))!=identity.get('seed'):raise ValueError('History seed is unknown/different')
    bound=receipt.get('verified_input_bindings',{}).get(key)
    if not bound or any(bound.get(k)!=identity.get(k)for k in ('image_sha256','arrays_sha256','anchor','instruction')):
        raise ValueError('Audited history original image/array/anchor/instruction binding absent')
    prediction=prediction_payload(row['result'],identity['model'])
    normalized_source=artifact.get('normalized_export_source_sha256')
    if normalized_source is not None:
        # Legacy schemas may be normalized by an independent root verifier,
        # but that export must still point to unchanged real output values.
        source_artifact=artifacts.get(normalized_source)
        approved=receipt.get('normalized_export_source_links',{}).get(origin['artifact_sha256'])
        projection=row.get('source_prediction_binding',{})
        source_status_accepted=(source_artifact is not None and(source_artifact.get('status')in('complete','complete_requested_inventory')
            or(source_artifact.get('status')=='partial'and normalized_source in receipt.get('independently_accepted_terminated_partial_run_artifacts',[]))))
        if (not source_artifact or approved!=normalized_source or not source_status_accepted
                or source_artifact.get('revision')!=identity['checkpoint_revision']
                or not artifact_model_matches(source_artifact, identity['model'])
                or projection.get('artifact_sha256')!=normalized_source):
            raise ValueError('Normalized legacy export lacks its actual independently audited source artifact')
        projected={}
        for field in prediction:
            spec=projection.get(field)
            if not isinstance(spec,dict)or not isinstance(spec.get('pointer'),list):
                raise ValueError('Legacy export actual prediction pointer is missing')
            value=resolve_pointer(source_artifact,spec['pointer'])
            if spec.get('wrap_single_step'):
                if identity['model']!='openvla':raise ValueError('Only native H1 horizon-axis wrapping is allowed')
                value=[value]
            projected[field]=value
        if prediction_payload(projected,identity['model'])!=prediction:
            raise ValueError('Normalized legacy export changed actual output values')
    if (canonical_sha(prediction)!=entry.get('prediction_sha256')
            or prediction_payload(entry['prediction'],identity['model'])!=prediction):
        raise ValueError('History predictions differ from the actual raw successful row')
    return key,prediction


def sync_directory(path):
    if os.name=='posix':
        fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(fd)
        finally:os.close(fd)


class VerifiedHistoryIndex:
    """Offset index, not an ever-growing in-RAM table of native predictions."""
    def __init__(self,manifest_path,expected_sha):
        self.manifest=sealed_json(manifest_path,expected_sha);self.index={};self.non_replayable={}
        if self.manifest.get('schema')!='verified_native_success_cache_manifest_v1':
            raise ValueError('Explicit verified history export manifest required')
        artifacts={r['sha256']:sealed_json(r['path'],r['sha256'])for r in self.manifest.get('artifacts',[])}
        receipts={r['sha256']:sealed_json(r['path'],r['sha256'])for r in self.manifest.get('verification_receipts',[])}
        for receipt in receipts.values():receipt['_verified_key_set']=set(receipt.get('verified_computation_keys',[]))
        self.entries_checked=0
        for descriptor in self.manifest.get('cache_files',[]):
            path=Path(descriptor['path'])
            if file_sha(path)!=descriptor['sha256']:raise ValueError('Historical cache export bytes differ')
            with path.open('rb')as stream:
                while True:
                    offset=stream.tell();line=stream.readline()
                    if not line:break
                    if not line.strip():raise ValueError('Blank history rows are not silently omitted')
                    entry=json.loads(line);key,_=verify_history_entry(entry,self.manifest,artifacts,receipts)
                    if key in self.index:raise ValueError('Duplicate historical computation requires explicit reconciliation')
                    self.index[key]=(path,offset,len(line),hashlib.sha256(line).hexdigest());self.entries_checked+=1
        for row in self.manifest.get('non_replayable_entries',[]):
            identity=row.get('computation_identity');key=row.get('computation_key')
            if (not isinstance(identity,dict)or canonical_sha(identity)!=key or row.get('status')not in('unknown','error')
                    or not is_sha(row.get('origin_artifact_sha256'))or row['origin_artifact_sha256']not in artifacts):
                raise ValueError('Prior unknown/failure exclusion lacks exact identity/artifact evidence')
            if key in self.index or key in self.non_replayable:raise ValueError('Conflicting history success/unknown requires root adjudication')
            self.non_replayable[key]=row
    def get(self,key,identity):
        item=self.index.get(key)
        if item is None:return None
        path,offset,length,expected=item
        with path.open('rb')as stream:stream.seek(offset);raw=stream.read(length)
        if hashlib.sha256(raw).hexdigest()!=expected:raise RuntimeError('Historical indexed cache changed')
        entry=json.loads(raw)
        if entry['computation_identity']!=identity:raise ValueError('History identity mismatch despite key')
        return prediction_payload(entry['prediction'],identity['model'])
    def blocked(self,key,identity):
        row=self.non_replayable.get(key)
        if row is not None and row['computation_identity']!=identity:raise ValueError('Prior exclusion identity differs')
        return row


class NewSuccessIndex:
    """Append native successes once; aliases read a bounded line by offset."""
    def __init__(self,path):self.journal=DurableJSONL(path);self.index={}
    def add(self,key,identity,prediction,request_id):
        if key in self.index:raise RuntimeError('Successful identical computation must alias, not overwrite')
        offset,length=self.journal.append('verified_native_success',key=key,identity=identity,prediction=prediction,request_id=request_id)
        self.index[key]=(offset,length,request_id)
    def get(self,key,identity):
        item=self.index.get(key)
        if item is None:return None
        offset,length,request_id=item
        with self.journal.path.open('rb')as stream:stream.seek(offset);entry=json.loads(stream.read(length))
        if entry['identity']!=identity:raise RuntimeError('Within-run success identity differs')
        return prediction_payload(entry['prediction'],identity['model']),request_id
    def close(self):self.journal.close()


def scan_terminal_prefix(path):
    """Linear read-only reconciliation after interruption; never truncate tails."""
    ids=set();counts={};actual=unknown=history=aliases=0;previous=None;sequence=0;tail=None
    with Path(path).open('rb')as stream:
        for raw in stream:
            try:
                row=json.loads(raw);audit=row.pop('audit_sha256')
                if not raw.endswith(b'\n')or canonical_sha(row)!=audit or row.get('previous_sha256')!=previous or row.get('sequence')!=sequence:
                    raise ValueError('Terminal journal chain is partial/corrupt')
                if row.get('event')!='request_terminal'or row['request_id']in ids:
                    raise ValueError('Terminal request is duplicated or malformed')
            except (ValueError,KeyError,TypeError)as error:
                tail={'first_invalid_offset':stream.tell()-len(raw),'error_type':type(error).__name__};break
            ids.add(row['request_id']);counts[row['status']]=counts.get(row['status'],0)+1
            actual+=row.get('actual_attempted')is True;unknown+=row.get('actual_attempted')is None
            history+=row['status']=='history_reuse';aliases+=row['status']=='within_run_alias'
            previous=audit;sequence+=1
    return {'request_ids':ids,'counts':counts,'actual_attempts':actual,'unknown_entries':unknown,
        'history_reuse_records':history,'within_run_alias_records':aliases,'valid_records':sequence,'preserved_invalid_tail':tail}
