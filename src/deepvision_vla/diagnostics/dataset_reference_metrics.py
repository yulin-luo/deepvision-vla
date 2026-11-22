"""Metric-only, sealed source-bound dataset references; never global physical GT.

This module does not load models or invent a producer transform. A root-audited
artifact must supply complete native-Hx7 targets and numeric provenance. Missing,
ambiguous or short references yield a named null result, never padded targets.
"""
import json
import math
import re
from pathlib import Path

from .formal_plan import is_sha
from .formal_execution_core import sealed_json,prediction_payload

HORIZONS={'openvla':1,'pi0':4,'qwen_h16':16}


class DatasetReferences:
    def __init__(self,path,expected_sha,codec_path,codec_sha):
        self.artifact=sealed_json(path,expected_sha);self.sha256=expected_sha
        self.codec=sealed_json(codec_path,codec_sha);self.codec_sha256=codec_sha
        if (self.artifact.get('schema')!='verified_source_bound_dataset_references_v1'
                or self.artifact.get('global_physical_gt_claim')is not False
                or self.artifact.get('model_outcomes_used_for_reference_selection')is not False):
            raise ValueError('Explicit independently verified dataset-reference artifact required')
        self.index={};self.ambiguous=set()
        for row in self.artifact.get('references',[]):
            key=(row.get('model'),row.get('sample_id'))
            if key in self.index:self.ambiguous.add(key)
            self.index[key]=row
        self.accepted=set(self.artifact.get('accepted_full_trajectory_numeric_proof_sha256',[]))
    def evaluate(self,prediction,source,model):
        base={'source_bound_dataset_reference_mse':None,'source_bound_dataset_reference_first_step_mse':None,
            'reference_artifact_sha256':self.sha256,'global_physical_gt_verified':False}
        reference=self.index.get((model,source['sample_id']))
        if (model,source['sample_id'])in self.ambiguous:return {**base,'dataset_reference_status':'ambiguous_duplicate_reference'}
        if reference is None:return {**base,'dataset_reference_status':'missing_verified_reference'}
        def absent(reason):return {**base,'dataset_reference_status':reason}
        if reference.get('status')!='verified_numeric_reference':return absent('reference_not_verified')
        if (reference.get('arrays_sha256')!=source.get('arrays_sha256')or reference.get('anchor')!=source.get('anchor')
                or reference.get('model_native_output_codec_sha256')!=self.codec_sha256):
            return absent('reference_input_or_model_output_contract_mismatch')
        proof=reference.get('full_trajectory_numeric_proof_sha256')
        if not is_sha(proof)or proof not in self.accepted:return absent('full_trajectory_numeric_proof_unaccepted')
        for field in ('action_definition_sha256','statistics_sha256'):
            if not is_sha(reference.get(field))or reference[field]!=self.codec.get(field):
                return absent('reference_'+field+'_unbound')
        if (not isinstance(reference.get('reference_dataset'),str)or not reference['reference_dataset']
                or not isinstance(reference.get('reference_revision'),str)
                or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',reference['reference_revision'])
                or type(reference.get('converted_episode_index'))is not int):
            return absent('reference_dataset_revision_episode_unknown')
        horizon=HORIZONS[model];frames=reference.get('converted_frame_indices')
        if (not isinstance(frames,list)or len(frames)!=horizon or any(type(x)is not int or x<0 for x in frames)
                or frames!=list(range(frames[0],frames[0]+horizon))):
            return absent('reference_short_or_ambiguous_window_no_padding')
        values=reference.get('native_reference_horizon')
        if (not isinstance(values,list)or len(values)!=horizon or any(not isinstance(row,list)or len(row)!=7 for row in values)
                or any(type(x)not in(int,float)or not math.isfinite(x)for row in values for x in row)):
            return absent('reference_full_native_horizon_missing_no_padding')
        native=prediction_payload(prediction,model)['native']
        errors=[(a-b)**2 for row,target in zip(native,values)for a,b in zip(row,target)]
        return {**base,'source_bound_dataset_reference_mse':sum(errors)/len(errors),
            'source_bound_dataset_reference_first_step_mse':sum(errors[:7])/7,
            'dataset_reference_status':'verified_source_bound_dataset_reference',
            'reference_dataset':reference['reference_dataset'],'reference_revision':reference['reference_revision'],
            'converted_episode_index':reference['converted_episode_index'],'converted_frame_indices':frames,
            'full_trajectory_numeric_proof_sha256':proof,'reference_horizon':horizon}
