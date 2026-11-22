"""Data-only fixed1500 request planning. No runner, acquisition or dispatch.

Reviewed AI rectangle proxies remain proxies: unverified physical contact and
the Pi/Qwen GT gates are never promoted by constructing a request manifest.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from .controls import matched_control_mask
from .geometry import VisualLayout,project_roi_boxes
from .roi_filter import filter_candidate_annotation
from .sampling import dropped_positions
from .random_controls import RANDOM_CONTROLS,prepare_random_control,validate_bindings

SOURCE_COUNT = 1500
LAYERS = {"openvla":(0,10,20,30),"pi0":(0,6,12,16),"qwen_h16":(0,12,24,34)}
REPORT_METRICS = {
    "openvla": ("official_bridge_orig_gt_mse",),
    "pi0": ("source_bound_H4_reference_mse",),
    "qwen_h16": ("native_H16_prediction_drift", "normalized_H16_prediction_drift"),
}
REVISIONS = {"openvla":"47a0ec7fc4ec123775a391911046cf33cf9ed83f",
    "pi0":"93c2840a2c68c337da0d472a2f52d1fd8d62132a","qwen_h16":"b131353d58b77daf2b6308e5ea4ea0aa4a417a79"}
KEEPS = tuple(i/10 for i in range(11))
PRIMARY = "primary_object_robot_proxy"
TASK_PRIMARY = "primary_task_roi"
TASK_ROI_SCHEMA = "formal1500_data_only_request_plan_v3"
OBJECT_ROBOT_BOX_ROLES = ("manipulated_object", "robot_gripper", "robot_arm")
TASK_ROI_BOX_ROLES = (*OBJECT_ROBOT_BOX_ROLES, "goal_context_anchor")
TASK_ROI_SCOPE = "visible_task_object_robot_interaction_rectangle_union"
PILOT_SOURCE_COUNT = 16
PILOT_KEEPS = (0., .5, 1.)
CONTROLS = ("roi_matched","non_target_object","empty_background")
ROLES = {*TASK_ROI_BOX_ROLES, *CONTROLS[1:]}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def is_sha(value):return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def _linked(value):
    return isinstance(value,dict) and isinstance(value.get("audit_link"),str) and bool(value["audit_link"].strip()) and is_sha(value.get("audit_sha256"))


def _size(source):
    size = source.get("image_size",[source.get("width"),source.get("height")])
    if not isinstance(size,(list,tuple)) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError("Explicit original image dimensions are required")
    if (source.get("width",size[0]),source.get("height",size[1])) != tuple(size):
        raise ValueError("Conflicting original image dimensions")
    return list(size)


def validate_inventory(inventory,expected_order_sha256):
    if len(inventory) != SOURCE_COUNT:raise ValueError("Exactly1500 original source identities are required")
    ids = [row.get("sample_id") for row in inventory]
    if any(not isinstance(s,str) or not s for s in ids) or len(set(ids)) != SOURCE_COUNT:
        raise ValueError("Original source IDs must be1500 distinct explicit identities")
    if canonical_sha(ids) != expected_order_sha256:raise ValueError("Original source ID order differs from its frozen seal")
    for ordinal,source in enumerate(inventory,1):
        if type(source.get("ordinal",ordinal)) is not int or source.get("ordinal",ordinal) != ordinal:
            raise ValueError("Source ordinal differs from immutable order")
        if not is_sha(source.get("image_sha256")):raise ValueError("Source image identity SHA is required")
        _size(source)
    return ids


def validate_pilot_ordinals(ordinals):
    if (not isinstance(ordinals, (list, tuple)) or len(ordinals) != PILOT_SOURCE_COUNT
            or any(type(i) is not int or not 1 <= i <= SOURCE_COUNT for i in ordinals)
            or list(ordinals) != sorted(set(ordinals))):
        raise ValueError("A runtime pilot requires 16 distinct original source ordinals in original order")
    return tuple(ordinals)


def _annotation(source,annotation,bank_revision):
    reasons = [];roles = [];rejected = []
    if annotation is None:return ["annotation_missing"],roles,rejected
    if (annotation.get("sample_id") != source["sample_id"] or annotation.get("image_sha256") != source["image_sha256"]
            or annotation.get("image_size") != _size(source) or annotation.get("coordinate_space") != "source_image_pixels"):
        reasons.append("annotation_source_identity_or_pixel_space_mismatch")
    if annotation.get("bank_revision") != bank_revision:reasons.append("annotation_bank_revision_mismatch")
    review = annotation.get("review",{})
    if (review.get("status") != "reviewed_pass" or review.get("method") != "ai_visual_review"
            or annotation.get("human_verified") is not False or review.get("human_verified") is not False):
        reasons.append("ai_review_not_qualified" if review.get("status") == "reviewed_blocked" else "ai_review_not_completed")
    if review.get("ambiguous") is not False:reasons.append("review_ambiguity_unresolved")
    if review.get("policy_results_used") is not False:reasons.append("outcome_independent_review_not_bound")
    if not _linked(review):reasons.append("review_audit_link_or_sha_missing")
    if review.get("source_row_sha256") != canonical_sha(source):reasons.append("review_source_row_unbound")
    raw_roles = annotation.get("roles")
    if not isinstance(raw_roles,list):reasons.append("role_annotations_missing");raw_roles = []
    if review.get("roles_sha256") != canonical_sha(raw_roles):reasons.append("review_role_geometry_unbound")
    if any(not isinstance(r,dict) or r.get("role") not in ROLES for r in raw_roles):
        reasons.append("unsupported_or_malformed_role")
    else:
        checked = filter_candidate_annotation({"sample_id":source["sample_id"],"image_sha256":source["image_sha256"],
            "image_size":_size(source),"coordinate_space":"source_image_pixels",
            "candidate_boxes":[{"xyxy_px":r.get("xyxy_px"),"annotation":r} for r in raw_roles]})
        roles = [r["annotation"] for r in checked["candidate_boxes"]];rejected = checked["rejected_candidates"]
        if rejected:reasons.append("source_role_box_invalid_or_greater_than60_percent")
    present = {r["role"] for r in roles}
    if "manipulated_object" not in present:reasons.append("manipulated_object_missing")
    if not present & {"robot_gripper","robot_arm"}:reasons.append("robot_arm_or_gripper_missing")
    goal = annotation.get("goal_region_status")
    if goal not in {"available","not_required"} or (goal == "available" and "goal_context_anchor" not in present):
        reasons.append("goal_context_missing_or_unresolved")
    goal_review = annotation.get("goal_requirement_review",{})
    if goal in {"available","not_required"}:
        if (not isinstance(goal_review,dict) or not _linked(goal_review)
                or goal_review.get("source_row_sha256") != canonical_sha(source)
                or not isinstance(goal_review.get("reason"),str) or not goal_review["reason"].strip()
                or goal_review.get("human_verified") is not False or goal_review.get("policy_results_used") is not False):
            reasons.append("goal_requirement_review_unbound")
        elif goal_review.get("spatial_target_required") is not (goal == "available"):
            reasons.append("goal_requirement_classification_conflicts_with_status")
    if annotation.get("physical_contact_claim") is not False:reasons.append("unsupported_physical_contact_claim")
    return reasons,roles,rejected


def _overlap(left,right):
    return min(left[2],right[2]) > max(left[0],right[0]) and min(left[3],right[3]) > max(left[1],right[1])


def pi_measured_input_verified(source, geometry):
    """Bind genuine native state passthrough; never infer a camera/world frame."""
    if not isinstance(geometry,dict):return False
    audit = geometry.get("state_input_audit",{})
    if not isinstance(audit,dict):return False
    measured = audit.get("measured_state7",[]);native = audit.get("native_state8_before_normalization",[])
    if (not isinstance(measured,list) or not isinstance(native,list) or len(measured) != 7 or len(native) != 8 or
            any(type(v) not in (int,float) or not math.isfinite(v) for v in measured+native)):
        return False
    return (is_sha(audit.get("source_array_sha256")) and audit.get("source_array_sha256") == source.get("arrays_sha256")
            and type(audit.get("original_anchor")) is int and audit["original_anchor"] == source.get("anchor")
            and audit.get("zero_pad_index") == 6 and audit.get("raw_gripper_destination_index") == 7
            and native == measured[:6]+[0.0]+measured[-1:]
            and audit.get("raw_gripper_preserved") is True
            and audit.get("gripper_clipped_or_binarized") is False
            and audit.get("zero_state_substitute_used") is False
            and audit.get("extra_axis_or_origin_transform_applied") is False
            and audit.get("physical_frame_origin_verified_by_preprocessing") is False)


class FormalPlanner:
    """Hold small source/evidence indices; generate bounded per-source rows."""
    def __init__(self,inventory,annotations,geometries,profiles,*,inventory_sha256,
                 expected_order_sha256,inventory_revision,bank_revision,include_controls=False,seed=42,
                 include_random_controls=False,random_control_metadata=None,random_control_bindings=None,
                 include_interaction_in_primary=False,pilot_source_ordinals=None):
        if not is_sha(inventory_sha256) or not inventory_revision or not bank_revision or seed != 42:
            raise ValueError("Explicit frozen inventory/bank revisions, SHA and seed42 are required")
        self.inventory = copy.deepcopy(list(inventory));ids = validate_inventory(self.inventory,expected_order_sha256)
        self.annotations = self._index(annotations,lambda r:r["sample_id"])
        self.geometries = self._index(geometries,lambda r:(r["sample_id"],r["model"]))
        if set(self.annotations)-set(ids) or {s for s,_ in self.geometries}-set(ids):
            raise ValueError("Foreign evidence cannot replace or add original identities")
        self.profiles = self._index(profiles,lambda r:r["identity"]["model"])
        if not self.profiles or set(self.profiles)-set(LAYERS):raise ValueError("Only the three fixed model families are supported")
        for model,profile in self.profiles.items():
            identity = profile["identity"]
            if identity.get("checkpoint_revision") != REVISIONS[model] or any(not is_sha(identity.get(k)) for k in (
                    "checkpoint_sha256","source_sha256","processor_sha256","action_mapping_sha256")):
                raise ValueError("Each actual model/checkpoint/source/processor/action identity must be sealed")
            if canonical_sha(profile.get("action_mapping",{})) != identity["action_mapping_sha256"]:
                raise ValueError("Action mapping fields differ from their sealed model identity")
            if model != "openvla" and profile.get("action_mapping",{}).get("gt_mapping_verified") is True:
                raise ValueError("Pi/Qwen GT mapping is unverified in this protocol; do not promote it")
            notes = profile.get("protocol_notes",[])
            if not isinstance(notes,list) or any(not isinstance(note,str) for note in notes):
                raise ValueError("Optional protocol notes are text metadata only")
        self.inventory_sha = inventory_sha256;self.order_sha = expected_order_sha256
        self.inventory_revision = inventory_revision;self.bank_revision = bank_revision
        self.include_controls = bool(include_controls);self.seed = seed
        self.evidence_input_sha256 = None
        if type(include_random_controls) is not bool:raise ValueError("Random controls require an explicit boolean opt-in")
        if type(include_interaction_in_primary) is not bool:
            raise ValueError("Interaction in the primary ROI requires an explicit boolean")
        if include_interaction_in_primary and include_random_controls:
            raise ValueError("Legacy paired random controls use a different primary ROI")
        self.include_interaction_in_primary = include_interaction_in_primary
        self.pilot_source_ordinals = None
        if pilot_source_ordinals is not None:
            if not include_interaction_in_primary or include_controls or include_random_controls:
                raise ValueError("The runtime pilot uses the primary task ROI only")
            self.pilot_source_ordinals = validate_pilot_ordinals(pilot_source_ordinals)
        self.keep_ratios = PILOT_KEEPS if self.pilot_source_ordinals else KEEPS
        self.primary_region = TASK_PRIMARY if include_interaction_in_primary else PRIMARY
        self.roi_scope = TASK_ROI_SCOPE if include_interaction_in_primary else "visible_object_robot_rectangle_union_proxy"
        self.include_random_controls=include_random_controls
        self.random_control_metadata={};self.random_control_bindings=None
        if include_random_controls:
            self.random_control_bindings=copy.deepcopy(validate_bindings(random_control_bindings,self.profiles))
            self.random_control_metadata=self._index(random_control_metadata or [],lambda r:(r['sample_id'],r['model']))
            if ({s for s,_ in self.random_control_metadata}-set(ids)
                    or {m for _,m in self.random_control_metadata}-set(LAYERS)):
                raise ValueError("Random metadata cannot replace/add original source or model identities")
        elif random_control_metadata is not None or random_control_bindings is not None:
            raise ValueError("Random metadata requires include_random_controls")

    @staticmethod
    def _index(rows,key):
        result = {}
        for row in rows:
            value = key(row)
            if value in result:raise ValueError("Duplicate evidence identity; do not choose a preferred copy")
            result[value] = copy.deepcopy(row)
        return result

    def _geometry(self,source,model):
        value = self.geometries.get((source["sample_id"],model));profile = self.profiles[model]
        if value is None:return None,["actual_model_geometry_missing"]
        if (value.get("actual_geometry_verified") is not True or value.get("geometry_source") != "actual_processor"
                or not _linked(value) or value.get("source_row_sha256") != canonical_sha(source)
                or value.get("model_identity_sha256") != canonical_sha(profile["identity"])
                or value.get("processor_sha256") != profile["identity"]["processor_sha256"]):
            return None,["actual_model_geometry_receipt_unbound"]
        try:
            data = dict(value["layout"])
            for name in ("source_size","crop_xyxy","canvas_size","content_xyxy","grid_shape","token_positions"):
                data[name] = tuple(data[name])
            layout = VisualLayout(**data)
            if layout.image_sha256 != source["image_sha256"] or list(layout.source_size) != _size(source):
                raise ValueError("Source geometry differs")
        except (KeyError,TypeError,ValueError):return None,["actual_model_geometry_invalid"]
        return layout,[]

    def _model_source(self,source,annotation,roles,base_reasons,model):
        profile = self.profiles[model];reasons = list(base_reasons)
        if not isinstance(source.get("instruction"),str) or not source["instruction"].strip():reasons.append("instruction_missing")
        mapping = profile.get("action_mapping",{})
        gt = model == "openvla" and mapping.get("gt_mapping_verified") is True and _linked(mapping)
        if model == "openvla" and not gt:reasons.append("gt_action_mapping_unverified")
        if model in {"openvla","pi0"} and (not is_sha(source.get("arrays_sha256"))
                or type(source.get("anchor")) is not int or source["anchor"] < 0):reasons.append("raw_action_state_reference_missing")
        layout,geometry_reasons = self._geometry(source,model);reasons += geometry_reasons
        if model == "pi0" and not pi_measured_input_verified(source,self.geometries.get((source["sample_id"],model),{})):
            reasons.append("measured_pi_native_input_unbound")
        primary = ();pools = {};control_reasons = [];role_token_counts = None
        if not reasons:
            def project(boxes):return project_roi_boxes(boxes,image_sha256=source["image_sha256"],layout=layout,membership="patch_center") if boxes else ()
            task_boxes = [r["xyxy_px"] for r in roles if r["role"] in OBJECT_ROBOT_BOX_ROLES]
            context_boxes = [r["xyxy_px"] for r in roles if r["role"] == "goal_context_anchor"]
            role_token_counts = {name:len(project([r["xyxy_px"] for r in roles if r["role"] == name])) for name in sorted(ROLES)}
            role_token_counts["robot_any"] = len(project([r["xyxy_px"] for r in roles if r["role"] in {"robot_arm","robot_gripper"}]))
            primary = project(task_boxes + context_boxes if self.include_interaction_in_primary else task_boxes)
            if not primary:reasons.append("primary_roi_has_no_actual_model_tokens")
            relevant = set(primary)|set(project(context_boxes));pools["roi_matched"] = primary
            controls = {}
            for region in CONTROLS[1:]:
                selected = [r for r in roles if r["role"] == region]
                controls[region] = [r["xyxy_px"] for r in selected]
                if not selected or any(r.get("control_qualification") != "reviewed_non_task" for r in selected):
                    control_reasons.append(region+"_not_qualified")
                if any(_overlap(box,task) for box in controls[region] for task in task_boxes+context_boxes):
                    control_reasons.append(region+"_overlaps_task_pixels")
                pools[region] = tuple(p for p in project(controls[region]) if p not in relevant)
            if any(_overlap(a,b) for a in controls["non_target_object"] for b in controls["empty_background"]):
                control_reasons.append("control_role_pixels_overlap")
            if set(pools["non_target_object"]) & set(pools["empty_background"]):control_reasons.append("control_role_tokens_overlap")
            if not all(pools.values()):control_reasons.append("control_pool_empty_after_actual_geometry")
        if reasons:control_reasons += reasons
        control_ok = not reasons and not control_reasons
        matched_size = min(map(len,pools.values())) if control_ok else 0
        matched = {role:matched_control_mask(pool,required_count=matched_size,sample_id=source["sample_id"],
                   region=role+"_formal_fixed_subpool",seed=self.seed)["positions"] for role,pool in pools.items()} if control_ok else {}
        return {"primary_reasons":sorted(set(reasons)),"control_reasons":sorted(set(control_reasons)),
            "primary_tokens":primary,"role_token_counts":role_token_counts,"control_pools":pools,"matched_pools":matched,"matched_size":matched_size,
            "primary_eligible":not reasons,"controls_eligible":control_ok,"gt_mse_allowed":gt,
            "geometry_receipt_sha256":canonical_sha(self.geometries.get((source["sample_id"],model)))}

    def source_plan(self,ordinal,source):
        annotation = self.annotations.get(source["sample_id"])
        reasons,roles,rejected = _annotation(source,annotation,self.bank_revision)
        requests = [];model_audits = {};source_hash = canonical_sha(source)
        for model,profile in self.profiles.items():
            metric = REPORT_METRICS[model][0] if self.include_interaction_in_primary else (
                "gt_action_mse" if model == "openvla" else "native_prediction_drift_not_gt")
            prepared = self._model_source(source,annotation,roles,reasons,model)
            random_prepared=None
            if self.include_random_controls:
                layout,_=self._geometry(source,model)
                random_prepared=prepare_random_control(self.random_control_metadata.get((source['sample_id'],model)),
                    source=source,annotation=annotation,roles=roles,layout=layout,
                    geometry=self.geometries.get((source['sample_id'],model)),profile=profile,
                    bindings=self.random_control_bindings,layers=LAYERS[model],primary_reasons=prepared['primary_reasons'])
            keys = set();eligible = 0
            definitions = [("baseline",None,None)] + [(self.primary_region,layer,keep) for layer in LAYERS[model] for keep in self.keep_ratios]
            if self.include_controls:definitions += [(region,layer,keep) for region in CONTROLS for layer in LAYERS[model] for keep in KEEPS]
            if self.include_random_controls:definitions += [(region,layer,keep) for region in RANDOM_CONTROLS for layer in LAYERS[model] for keep in KEEPS]
            cohort_counts={name:Counter() for name in ('primary','semantic','paired_random_visual')}
            cohort_keys={name:set() for name in cohort_counts}
            for index,(region,layer,keep) in enumerate(definitions):
                primary_request = region in {"baseline",self.primary_region}
                random_request=region in RANDOM_CONTROLS
                ready = random_prepared['eligible'] if random_request else prepared["primary_eligible"] if primary_request else prepared["controls_eligible"]
                blocked = random_prepared['reasons'] if random_request else prepared["primary_reasons"] if primary_request else prepared["control_reasons"]
                pool_size = random_prepared['k'] if random_request else None if region == "baseline" else len(prepared["primary_tokens"]) if primary_request else prepared["matched_size"]
                positions = None
                if ready:
                    if region == "baseline":positions = []
                    elif region == self.primary_region:positions = list(dropped_positions(prepared["primary_tokens"],keep_ratio=keep,sample_id=source["sample_id"],layer=layer,seed=self.seed))
                    elif random_request:positions=list(random_prepared['masks'][(region,round(keep*10))])
                    else:
                        count = pool_size-round(pool_size*keep)
                        positions = list(matched_control_mask(prepared["matched_pools"][region],required_count=count,
                            sample_id=source["sample_id"],region=region+"_formal_mask_order",seed=self.seed)["positions"])
                cache_key = canonical_sha([source_hash,profile["identity"],self.bank_revision,
                    prepared["geometry_receipt_sha256"],layer,positions,self.seed]) if ready else None
                if ready:eligible += 1;keys.add(cache_key)
                dropped = len(positions) if positions is not None else None
                retained = pool_size-dropped if pool_size and dropped is not None else None
                request={"request_id":f"{ordinal:04d}:{source['sample_id']}:{model}:{index}",
                    "ordinal":ordinal,"sample_id":source["sample_id"],"source_row_sha256":source_hash,
                    "image_sha256":source["image_sha256"],"model":model,"model_identity_sha256":canonical_sha(profile["identity"]),
                    "region":region,"layer":layer,"keep_ratio":keep,"positions":positions,"seed":self.seed,
                    "status":"eligible_proxy" if ready else "blocked","blocked_reasons":list(blocked),
                    "pool_token_count":pool_size,"distinct_dropped_tokens":dropped,"retained_token_count":retained,
                    "actual_retained_fraction":retained/pool_size if retained is not None else None,
                    "success_cache_key":cache_key,"cache_reuse_requires_prior_verified_success":True,
                    "gt_mse_allowed":prepared["gt_mse_allowed"] if ready else False,
                    "metric":metric,
                    "roi_scope":self.roi_scope,"physical_contact_claim":False,
                    "human_verified":False,"paper_protocol_blocked_reasons":["contact_geometry_unverified","human_verification_absent"],
                    "paper_reproduction":False,"launch_allowed":False,"actual_attempted":False}
                if self.include_random_controls:
                    cohort='paired_random_visual' if random_request else 'primary' if primary_request else 'semantic'
                    request['request_cohort']=cohort;cohort_counts[cohort]['requested']+=1
                    cohort_counts[cohort]['eligible_proxy_requests' if ready else 'blocked_requests']+=1
                    if ready:cohort_keys[cohort].add(cache_key)
                    if random_request:
                        request['semantic_background_claim']=False
                        request['roi_scope']='paired_random_visual_control_not_semantic_background'
                        request['random_control_provenance']={
                            'metadata_file_sha256':self.random_control_bindings['metadata_sha256'],
                            'protocol_sha256':self.random_control_bindings['protocol_sha256'],
                            'metadata_row_sha256':random_prepared['metadata_row_sha256'],
                            'strict_pixel_disjoint_pool_sha256':random_prepared['strict_pool_sha256']}
                requests.append(request)
            model_audits[model] = {"requested":len(definitions),"eligible_proxy_requests":eligible,"blocked_requests":len(definitions)-eligible,
                "prospective_distinct_computations":len(keys),"potential_alias_output_points":eligible-len(keys),
                "primary_source_eligible":prepared["primary_eligible"],"controls_source_eligible":prepared["controls_eligible"],
                "primary_blocked_reasons":prepared["primary_reasons"],"control_blocked_reasons":prepared["control_reasons"],
                "primary_token_count":len(prepared["primary_tokens"]),"role_token_counts":prepared["role_token_counts"],"matched_subpool_size":prepared["matched_size"],
                "geometry_receipt_sha256":prepared["geometry_receipt_sha256"],"gt_mse_allowed":prepared["gt_mse_allowed"]}
            if self.include_random_controls:
                model_audits[model].update(random_controls_source_eligible=random_prepared['eligible'],
                    random_control_blocked_reasons=random_prepared['reasons'],random_matched_subpool_size=random_prepared['k'],
                    random_metadata_proposed_K=random_prepared['metadata_proposed_K'],
                    random_metadata_row_sha256=random_prepared['metadata_row_sha256'],
                    control_cohorts={name:{**{key:c[key] for key in ('requested','eligible_proxy_requests','blocked_requests')},'prospective_distinct_computations':len(cohort_keys[name]),
                        'eligible_sources':int(c['eligible_proxy_requests']>0),'blocked_sources':int(c['requested']>0 and c['eligible_proxy_requests']==0),
                        'potential_alias_output_points':c['eligible_proxy_requests']-len(cohort_keys[name])}
                        for name,c in cohort_counts.items()})
        return {"ordinal":ordinal,"sample_id":source["sample_id"],"source_row_sha256":source_hash,
            "annotation_sha256":canonical_sha(annotation),"source_blocked_reasons":reasons,"rejected_role_boxes":rejected,
            "human_verified":False,"physical_contact_claim":False,"models":model_audits,"requests":requests}

    def iter_sources(self):
        ordinals = self.pilot_source_ordinals or range(1, SOURCE_COUNT + 1)
        for ordinal in ordinals:
            source = self.source_plan(ordinal, self.inventory[ordinal - 1])
            if self.pilot_source_ordinals and not all(m["primary_source_eligible"] for m in source["models"].values()):
                raise ValueError(f"Runtime pilot source {ordinal} lacks qualified primary geometry or review")
            yield source

    def manifest(self):
        protocol_notes = {m:{"intervention":"single_layer_input_intervention",
            "native_backbone_branches_changed":False,"additional_profile_notes":p.get("protocol_notes",[])}
            for m,p in self.profiles.items()}
        if "qwen_h16" in protocol_notes:
            protocol_notes["qwen_h16"].update(deepstack_branch_unchanged=True,
                native_deepstack_reinjection_after_layers=[0,1,2],layer0_is_not_all_visual_information_ablation=True,
                interpretation="L12 versus L0 can include native reinjection differences; this is a potential confound, not an established cause")
        manifest={"schema":"formal1500_data_only_request_plan_v1","inventory_sha256":self.inventory_sha,
            "original_id_order_sha256":self.order_sha,"inventory_revision":self.inventory_revision,"bank_revision":self.bank_revision,
            "requested_source_identities":SOURCE_COUNT,"models":list(self.profiles),"paper_layers":{m:list(LAYERS[m]) for m in self.profiles},
            "keep_ratios":list(self.keep_ratios),"primary_requests_per_source_model":1+4*len(self.keep_ratios),"optional_control_requests_per_source_model":132 if self.include_controls else 0,
            "include_controls":self.include_controls,"membership":"patch_center","per_source_box_area_limit":.6,
            "area_rejection":"strictly_greater_than_individual_box; never_clip_or_limit_union",
            "mask_order":"nested_by_source_region_seed_shared_across_layers","seed":self.seed,
            "component_coverage":"Per-role token-center counts reported; nonempty union does not imply each small object contains a token center; no nearest-token fallback",
            "model_protocol_notes":protocol_notes,
            "cache":"prospective_only; identical input/model/layer/mask/seed may reuse verified successes, never failures",
            "human_verified":False,"physical_contact_claim":False,"paper_reproduction":False,"launch_allowed":False,
            "model_calls":0,"gpu_calls":0,"optimizer_steps":0,"geometry_reexecuted":False}
        if self.include_random_controls:
            manifest.update(schema='formal1500_data_only_request_plan_v2',include_random_controls=True,
                optional_random_control_requests_per_source_model=88,random_control_regions=list(RANDOM_CONTROLS),
                random_control_evidence=self.random_control_bindings,
                random_control_membership='primary_patch_center; random_valid_full_content_cells_exclude_positive_pixel_intersection_object_robot_goal',
                random_control_semantic_background_claim=False,
                cohort_request_budgets={'primary':45,'semantic':132 if self.include_controls else 0,'paired_random_visual':88})
        if self.include_interaction_in_primary:
            manifest.update(schema=TASK_ROI_SCHEMA, primary_region=TASK_PRIMARY,
                            primary_role_union=list(TASK_ROI_BOX_ROLES),
                            report_metrics={model: list(REPORT_METRICS[model]) for model in self.profiles})
        if self.pilot_source_ordinals:
            manifest.update(pilot_source_ordinals=list(self.pilot_source_ordinals),
                            eligible_for_population_report=False)
        return manifest


def write_plan(planner,output):
    """Stream bounded source blocks; never rewrite a growing progress array."""
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    counts = {model:Counter() for model in planner.profiles};request_hash = hashlib.sha256();audit_hash = hashlib.sha256();seen = 0
    cohort_totals={model:{name:Counter() for name in ('primary','semantic','paired_random_visual')} for model in planner.profiles}
    k_histograms={model:{name:Counter() for name in ('semantic','paired_random_visual')} for model in planner.profiles}
    with (output/"requests.jsonl").open("xb") as stream,(output/"source_audit.jsonl").open("xb") as audit:
        for result in planner.iter_sources():
            seen += 1
            for row in result.pop("requests"):
                raw = (json.dumps(row,separators=(",",":"),allow_nan=False)+"\n").encode();stream.write(raw);request_hash.update(raw)
            raw_audit = (json.dumps(result,separators=(",",":"),allow_nan=False)+"\n").encode();audit.write(raw_audit);audit_hash.update(raw_audit)
            for model,model_audit in result["models"].items():
                for key in ("requested","eligible_proxy_requests","blocked_requests","prospective_distinct_computations","potential_alias_output_points",
                            "primary_source_eligible","controls_source_eligible"):counts[model][key] += model_audit[key]
                if planner.include_random_controls:
                    for name,cohort in model_audit['control_cohorts'].items():cohort_totals[model][name].update(cohort)
                    if planner.include_controls and model_audit['controls_source_eligible']:
                        k_histograms[model]['semantic'][str(model_audit['matched_subpool_size'])]+=1
                    if model_audit['random_controls_source_eligible']:
                        k_histograms[model]['paired_random_visual'][str(model_audit['random_matched_subpool_size'])]+=1
    expected_sources = len(planner.pilot_source_ordinals) if planner.pilot_source_ordinals else SOURCE_COUNT
    primary_requests = 1 + 4*len(planner.keep_ratios)
    per_model = expected_sources*(primary_requests+(132 if planner.include_controls else 0)+(88 if planner.include_random_controls else 0))
    if seen != expected_sources or any(c["requested"] != per_model for c in counts.values()):
        raise RuntimeError("Partial request stream cannot publish a complete workload manifest")
    manifest = {**planner.manifest(),"evidence_input_sha256":planner.evidence_input_sha256,
                "synthetic_source_count":sum(s.get("synthetic_test_fixture") is True for s in planner.inventory),
                "prospective_workload":{m:dict(c) for m,c in counts.items()},
                "requests_jsonl_sha256":request_hash.hexdigest(),"source_audit_jsonl_sha256":audit_hash.hexdigest()}
    if planner.include_random_controls:
        manifest['prospective_control_cohorts']={model:{name:dict(value) for name,value in cohorts.items()} for model,cohorts in cohort_totals.items()}
        manifest['matched_K_histograms']={model:{name:dict(value) for name,value in histograms.items()} for model,histograms in k_histograms.items()}
    with (output/"manifest.json").open("x",encoding="utf8") as stream:json.dump(manifest,stream,indent=2,allow_nan=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory","annotations","geometries","profiles","output"):parser.add_argument("--"+name,required=True,type=Path)
    for name in ("expected-inventory-sha256","expected-id-order-sha256","inventory-revision","bank-revision"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--include-controls",action="store_true")
    parser.add_argument("--include-interaction-in-primary", action="store_true",
                        help="Use reviewed object, robot and interaction boxes in the task ROI")
    parser.add_argument("--pilot-source-ordinals", type=Path,
                        help="JSON array of 16 approved source ordinals for a runtime-only pilot")
    parser.add_argument("--include-random-controls",action="store_true")
    parser.add_argument("--random-control-metadata",type=Path)
    parser.add_argument("--random-control-bindings",type=Path)
    parser.add_argument("--expected-random-control-bindings-sha256")
    args = parser.parse_args()
    inventory_raw = args.inventory.read_bytes();actual_sha = hashlib.sha256(inventory_raw).hexdigest()
    if actual_sha != args.expected_inventory_sha256:raise ValueError("Original inventory byte seal differs")
    evidence = {name:getattr(args,name).read_bytes() for name in ("annotations","geometries","profiles")}
    rows = lambda raw:json.loads(raw) if raw.lstrip().startswith(b"[") else [json.loads(line) for line in raw.decode("utf8").splitlines() if line.strip()]
    random_kwargs={}
    if args.include_random_controls:
        if not args.random_control_metadata or not args.random_control_bindings or not is_sha(args.expected_random_control_bindings_sha256):
            raise ValueError("Opt-in random controls require sealed metadata/bindings paths")
        binding_raw=args.random_control_bindings.read_bytes()
        if hashlib.sha256(binding_raw).hexdigest()!=args.expected_random_control_bindings_sha256:raise ValueError("Random binding bytes differ")
        binding=json.loads(binding_raw);metadata_raw=args.random_control_metadata.read_bytes()
        if (hashlib.sha256(metadata_raw).hexdigest()!=binding.get('metadata_sha256')
                or hashlib.sha256(evidence['annotations']).hexdigest()!=binding.get('source_bank_sha256')):
            raise ValueError("Random metadata/source-bank byte seals differ")
        random_kwargs={'include_random_controls':True,'random_control_metadata':rows(metadata_raw),'random_control_bindings':binding}
    elif args.random_control_metadata or args.random_control_bindings or args.expected_random_control_bindings_sha256:
        raise ValueError("Random metadata paths require explicit opt-in")
    planner = FormalPlanner(rows(inventory_raw),rows(evidence["annotations"]),rows(evidence["geometries"]),
        json.loads(evidence["profiles"].decode("utf8")),inventory_sha256=actual_sha,
        expected_order_sha256=args.expected_id_order_sha256,inventory_revision=args.inventory_revision,
        bank_revision=args.bank_revision,include_controls=args.include_controls,
        include_interaction_in_primary=args.include_interaction_in_primary,
        pilot_source_ordinals=json.loads(args.pilot_source_ordinals.read_bytes()) if args.pilot_source_ordinals else None,
        **random_kwargs)
    planner.evidence_input_sha256 = {name:hashlib.sha256(raw).hexdigest() for name,raw in evidence.items()}
    result = write_plan(planner,args.output)
    print(json.dumps({"launch_allowed":False,"source_identities":len(planner.pilot_source_ordinals or planner.inventory),
                      "prospective_workload":result["prospective_workload"]}))


if __name__ == "__main__":main()
