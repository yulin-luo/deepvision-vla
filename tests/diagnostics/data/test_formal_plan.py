"""Synthetic1500-ID contracts only: these are not real reviewed annotations."""
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch

from deepvision_vla.diagnostics import formal_plan as formal
from deepvision_vla.diagnostics.formal_execution_core import validate_manifest, validate_request_rows
from deepvision_vla.diagnostics.geometry import VisualLayout


def fixtures(models=("openvla",),controls=False,reviewed=True):
    sources = [{"ordinal":i+1,"sample_id":f"{i:020x}","image_sha256":hashlib.sha256(f"synthetic-image-{i}".encode()).hexdigest(),
                "image_size":[100,100],"instruction":"pick object","arrays_sha256":"c"*64,"anchor":0,
                "state_frame":"bridge_dataset_topdown","synthetic_test_fixture":True} for i in range(1500)]
    profiles = []
    for model in models:
        mapping = {"gt_mapping_verified":model == "openvla","audit_link":"synthetic-fixture://mapping",
                   "audit_sha256":"d"*64,"contract_name":"synthetic_test_only"}
        profiles.append({"identity":{"model":model,"checkpoint_revision":formal.REVISIONS[model],
            "checkpoint_sha256":"a"*64,"source_sha256":"b"*64,"processor_sha256":"e"*64,
            "action_mapping_sha256":formal.canonical_sha(mapping)},"action_mapping":mapping})
    source = sources[0]
    roles = [{"role":"manipulated_object","xyxy_px":[0,0,25,25]},
             {"role":"robot_gripper","xyxy_px":[25,0,50,25]},
             {"role":"non_target_object","xyxy_px":[0,50,50,75],"control_qualification":"reviewed_non_task"},
             {"role":"empty_background","xyxy_px":[50,50,100,75],"control_qualification":"reviewed_non_task"}]
    annotation = {"sample_id":source["sample_id"],"image_sha256":source["image_sha256"],"image_size":[100,100],
        "coordinate_space":"source_image_pixels","bank_revision":"synthetic-bank-v1","human_verified":False,
        "physical_contact_claim":False,"contact_annotation_status":"unverified","goal_region_status":"not_required","roles":roles,
        "review":{"status":"reviewed_pass" if reviewed else "unreviewed","method":"ai_visual_review","human_verified":False,
            "ambiguous":False,"policy_results_used":False,"audit_link":"synthetic-fixture://review","audit_sha256":"f"*64,
            "source_row_sha256":formal.canonical_sha(source),"roles_sha256":formal.canonical_sha(roles)}}
    annotation["goal_requirement_review"] = {"source_row_sha256":formal.canonical_sha(source),
        "reason":"Synthetic fixture: no distinct spatial destination","spatial_target_required":False,
        "audit_link":"synthetic-fixture://goal-requirement","audit_sha256":"8"*64,
        "human_verified":False,"policy_results_used":False}
    geometry = []
    for profile in profiles:
        model = profile["identity"]["model"]
        layout = VisualLayout(source["image_sha256"],(100,100),(0,0,100,100),(100,100),(0,0,100,100),(4,4),tuple(range(1,17)))
        geometry.append({"sample_id":source["sample_id"],"model":model,"actual_geometry_verified":True,
            "geometry_source":"actual_processor","audit_link":"synthetic-fixture://geometry","audit_sha256":"9"*64,
            "source_row_sha256":formal.canonical_sha(source),"model_identity_sha256":formal.canonical_sha(profile["identity"]),
            "processor_sha256":profile["identity"]["processor_sha256"],"layout":asdict(layout)})
    for record in geometry:
        if record["model"] == "pi0":
            measured = [.1,.2,.3,.4,.5,.6,1.00015]
            record["state_input_audit"] = {"measured_state7":measured,"native_state8_before_normalization":measured[:6]+[0.0]+measured[-1:],
                "source_array_sha256":source["arrays_sha256"],"original_anchor":source["anchor"],"zero_pad_index":6,
                "raw_gripper_destination_index":7,"raw_gripper_preserved":True,"gripper_clipped_or_binarized":False,
                "zero_state_substitute_used":False,"extra_axis_or_origin_transform_applied":False,
                "physical_frame_origin_verified_by_preprocessing":False}
    return sources,[annotation],geometry,profiles,{"inventory_sha256":formal.canonical_sha(sources),
        "expected_order_sha256":formal.canonical_sha([s["sample_id"] for s in sources]),
        "inventory_revision":"synthetic-inventory-v1","bank_revision":"synthetic-bank-v1","include_controls":controls}


def planner(parts):return formal.FormalPlanner(*parts[:4],**parts[4])


class FormalPlanContracts(unittest.TestCase):
    def test_runtime_pilot_keeps_original_ordinals_and_refuses_incomplete_evidence(self):
        sources, bank, geometry, profiles, options = fixtures()
        ordinals = [n * 77 for n in range(1,17)]
        for n in ordinals:
            source = sources[n - 1]
            row = copy.deepcopy(bank[0])
            row.update(sample_id=source["sample_id"], image_sha256=source["image_sha256"])
            for key in ("review", "goal_requirement_review"):
                row[key]["source_row_sha256"] = formal.canonical_sha(source)
            bank.append(row)
            item = copy.deepcopy(geometry[0])
            item.update(sample_id=source["sample_id"], source_row_sha256=formal.canonical_sha(source))
            item["layout"]["image_sha256"] = source["image_sha256"]
            geometry.append(item)
        options.update(include_interaction_in_primary=True, pilot_source_ordinals=ordinals)
        current = formal.FormalPlanner(sources, bank, geometry, profiles, **options)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"pilot"
            manifest = formal.write_plan(current, output)
            self.assertEqual(manifest["pilot_source_ordinals"], ordinals)
            self.assertFalse(manifest["eligible_for_population_report"])
            self.assertEqual(manifest["requested_source_identities"], 1500)
            self.assertEqual(manifest["keep_ratios"], [0.,.5,1.])
            requests = [json.loads(line) for line in (output/"requests.jsonl").read_bytes().splitlines()]
            self.assertEqual([row["ordinal"] for row in requests[::13]], ordinals)
            executable_metadata = {**manifest, "synthetic_source_count":0}
            counts = validate_request_rows(requests, executable_metadata, sources, profiles[0])
            self.assertEqual((counts["sources"],counts["requested"],counts["eligible"]),(16,208,208))
            with self.assertRaisesRegex(ValueError,"Partial plan"):
                validate_request_rows(requests[:-1], executable_metadata, sources, profiles[0])
            with self.assertRaisesRegex(ValueError,"order/coverage"):
                validate_request_rows(requests[13:], executable_metadata, sources, profiles[0])
            with self.assertRaisesRegex(ValueError,"requires 16 distinct"):
                validate_manifest({**executable_metadata,"pilot_source_ordinals":ordinals[:15]+[ordinals[0]]},"openvla")
        with self.assertRaisesRegex(ValueError,"lacks qualified primary"):
            incomplete = formal.FormalPlanner(sources, bank[:-1], geometry, profiles, **options)
            list(incomplete.iter_sources())

    def test_task_roi_adds_reviewed_interaction_without_changing_legacy_masks(self):
        parts = fixtures(controls=True)
        annotation = parts[1][0]
        annotation["roles"].append({"role":"goal_context_anchor","xyxy_px":[50,0,75,25]})
        annotation["goal_region_status"] = "available"
        annotation["goal_requirement_review"]["spatial_target_required"] = True
        annotation["review"]["roles_sha256"] = formal.canonical_sha(annotation["roles"])
        previous = planner(parts)
        self.assertEqual(previous.source_plan(1, parts[0][0])["requests"][1]["positions"], [1,2])
        self.assertNotIn("primary_role_union", previous.manifest())

        parts[4]["include_interaction_in_primary"] = True
        updated = planner(parts)
        manifest = {**updated.manifest(), "synthetic_source_count":0}
        self.assertEqual(manifest["schema"], formal.TASK_ROI_SCHEMA)
        report = updated.source_plan(1, parts[0][0])
        self.assertEqual(manifest["report_metrics"], {"openvla": ["official_bridge_orig_gt_mse"]})
        self.assertEqual(report["requests"][0]["metric"], "official_bridge_orig_gt_mse")
        self.assertEqual(report["requests"][1]["region"], formal.TASK_PRIMARY)
        self.assertEqual(report["requests"][1]["positions"], [1,2,3])
        self.assertEqual(report["requests"][11]["positions"], [])
        self.assertEqual(report["requests"][12]["positions"], [1,2,3])
        self.assertEqual(report["models"]["openvla"]["matched_subpool_size"], 2)

        invalid = {**manifest, "primary_role_union":list(formal.TASK_ROI_BOX_ROLES[:-1])}
        with self.assertRaisesRegex(ValueError, "explicit object, robot and interaction"):
            validate_manifest(invalid, "openvla")
        with self.assertRaisesRegex(ValueError, "scope differs"):
            invalid_requests = [{**report["requests"][0], "roi_scope":"legacy"}]
            validate_request_rows(invalid_requests, manifest, updated.inventory, parts[3][0])
        with self.assertRaisesRegex(ValueError, "Primary report metric differs"):
            invalid_requests = [{**report["requests"][0], "metric":"native_H16_prediction_drift"}]
            validate_request_rows(invalid_requests, manifest, updated.inventory, parts[3][0])
        all_requests = (request for source in updated.iter_sources() for request in source["requests"])
        validation = validate_request_rows(all_requests, manifest, updated.inventory, parts[3][0])
        self.assertEqual(validation["requested"], 1500 * 177)
        self.assertEqual(validation["eligible"], 177)

    def test_task_roi_pi_and_qwen_declare_distinct_metric_references(self):
        parts = fixtures(models=("pi0", "qwen_h16"))
        parts[4]["include_interaction_in_primary"] = True
        current = planner(parts)
        manifest = {**current.manifest(), "synthetic_source_count": 0}
        self.assertEqual(manifest["report_metrics"], {
            "pi0": ["source_bound_H4_reference_mse"],
            "qwen_h16": ["native_H16_prediction_drift", "normalized_H16_prediction_drift"],
        })
        for row in current.source_plan(1, current.inventory[0])["requests"]:
            self.assertEqual(row["metric"], manifest["report_metrics"][row["model"]][0])
            self.assertFalse(row["gt_mse_allowed"])
        legacy = {key: value for key, value in manifest.items() if key != "report_metrics"}
        self.assertEqual(validate_manifest(legacy, "pi0"), 45)
        with self.assertRaisesRegex(ValueError, "Report metrics differ"):
            validate_manifest({**manifest, "report_metrics": {"pi0": ["official_bridge_orig_gt_mse"]}}, "pi0")

    def test_task_roi_keeps_explicit_goal_availability_and_legacy_random_protocol(self):
        parts = fixtures()
        parts[4]["include_interaction_in_primary"] = True
        current = planner(parts)
        self.assertEqual(current.source_plan(1, parts[0][0])["requests"][1]["positions"], [1,2])

        parts[1][0]["goal_region_status"] = "unresolved"
        current = planner(parts)
        missing = current.source_plan(1, parts[0][0])["requests"][1]
        self.assertEqual(missing["status"], "blocked")
        self.assertIn("goal_context_missing_or_unresolved", missing["blocked_reasons"])

        parts[4]["include_random_controls"] = True
        with self.assertRaisesRegex(ValueError, "different primary ROI"):
            planner(parts)

    def test_cli_consumes_staged_json_array_inventory_and_jsonl_evidence(self):
        parts=fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);paths={k:root/(k+".json") for k in ("inventory","annotations","geometries","profiles")}
            paths["inventory"].write_text(json.dumps(parts[0],indent=2),encoding="utf8")
            paths["profiles"].write_text(json.dumps(parts[3]),encoding="utf8")
            for key,index in [("annotations",1),("geometries",2)]:
                paths[key].write_text("".join(json.dumps(v)+"\n" for v in parts[index]),encoding="utf8")
            actual=hashlib.sha256(paths["inventory"].read_bytes()).hexdigest()
            args=["formal-plan"]
            for key,value in paths.items():args += ["--"+key,str(value)]
            args += ["--output",str(root/"out"),"--expected-inventory-sha256",actual,
                "--expected-id-order-sha256",parts[4]["expected_order_sha256"],
                "--inventory-revision","synthetic-inventory-v1","--bank-revision","synthetic-bank-v1"]
            with patch("sys.argv",args),patch.object(formal,"write_plan",return_value={"prospective_workload":{}}) as write:
                formal.main()
            plan=write.call_args.args[0]
            self.assertEqual(plan.inventory,parts[0]);self.assertEqual(plan.inventory_sha,actual)
            self.assertEqual(len(plan.annotations),1);self.assertEqual(len(plan.geometries),1)

    def test_exact1500_unique_identity_order_is_sealed(self):
        parts = fixtures()
        for change in ("short","duplicate","reverse"):
            modified = copy.deepcopy(parts)
            if change == "short":modified[0].pop()
            elif change == "duplicate":modified[0][-1]["sample_id"] = modified[0][0]["sample_id"]
            else:modified[0].reverse()
            with self.assertRaises(ValueError):planner(modified)

    def test_synthetic1500_full_primary_denominator_preserved_when_unreviewed(self):
        parts = fixtures(reviewed=False);parts[1].clear();parts[2].clear()
        total = blocked = sources = 0;last = None
        for result in planner(parts).iter_sources():
            sources += 1;last = result["sample_id"]
            total += len(result["requests"])
            blocked += sum(r["status"] == "blocked" for r in result["requests"])
            self.assertTrue(all(r["launch_allowed"] is False and r["actual_attempted"] is False for r in result["requests"]))
        self.assertEqual((sources,total,blocked),(1500,67500,67500))
        self.assertEqual(last,parts[0][-1]["sample_id"])

    def test_reviewed_proxy45_and_contact_flags_never_become_paper_ready(self):
        p = planner(fixtures());result = p.source_plan(1,p.inventory[0]);rows = result["requests"]
        self.assertEqual(len(rows),45)
        self.assertTrue(all(r["status"] == "eligible_proxy" for r in rows))
        self.assertTrue(all(r["human_verified"] is False and r["physical_contact_claim"] is False and r["paper_reproduction"] is False for r in rows))
        self.assertTrue(all("contact_geometry_unverified" in r["paper_protocol_blocked_reasons"] for r in rows))
        self.assertFalse(p.manifest()["launch_allowed"])
        self.assertEqual(p.manifest()["model_calls"],0)

    def test_reviewed_robot_arm_is_preserved_as_arm_not_relabelled_gripper(self):
        parts = fixtures();annotation = parts[1][0]
        annotation["roles"][1]["role"] = "robot_arm"
        annotation["review"]["roles_sha256"] = formal.canonical_sha(annotation["roles"])
        reasons,roles,_ = formal._annotation(parts[0][0],annotation,"synthetic-bank-v1")
        self.assertEqual(reasons,[])
        self.assertIn("robot_arm",{r["role"] for r in roles})
        self.assertNotIn("robot_gripper",{r["role"] for r in roles})
        p = planner(parts);rows = p.source_plan(1,p.inventory[0])["requests"]
        self.assertTrue(all(r["status"] == "eligible_proxy" for r in rows))
        annotation["roles"] = [r for r in annotation["roles"] if r["role"] != "robot_arm"]
        annotation["review"]["roles_sha256"] = formal.canonical_sha(annotation["roles"])
        reasons,_,_ = formal._annotation(parts[0][0],annotation,"synthetic-bank-v1")
        self.assertIn("robot_arm_or_gripper_missing",reasons)

    def test_component_token_coverage_reports_tiny_object_without_nearest_fallback(self):
        parts=fixtures();annotation=parts[1][0]
        annotation["roles"][0]["xyxy_px"]=[1,1,2,2]
        annotation["review"]["roles_sha256"]=formal.canonical_sha(annotation["roles"])
        p=planner(parts);result=p.source_plan(1,p.inventory[0]);audit=result["models"]["openvla"]
        self.assertTrue(audit["primary_source_eligible"])
        self.assertEqual(audit["role_token_counts"]["manipulated_object"],0)
        self.assertGreater(audit["role_token_counts"]["robot_any"],0)
        blocked=planner(fixtures(reviewed=False));result=blocked.source_plan(1,blocked.inventory[0])
        self.assertIsNone(result["models"]["openvla"]["role_token_counts"])

    def test_optional_controls177_with_same_distinct_token_budget(self):
        p = planner(fixtures(controls=True));result = p.source_plan(1,p.inventory[0]);rows = result["requests"]
        self.assertEqual(len(rows),177)
        for layer in formal.LAYERS["openvla"]:
            for keep in formal.KEEPS:
                matched = [r for r in rows if r["region"] in formal.CONTROLS and r["layer"] == layer and r["keep_ratio"] == keep]
                self.assertEqual(len(matched),3)
                self.assertEqual(len({r["distinct_dropped_tokens"] for r in matched}),1)
        audit = result["models"]["openvla"]
        keys = {r["success_cache_key"] for r in rows}
        self.assertEqual(audit["prospective_distinct_computations"],len(keys))
        self.assertEqual(audit["potential_alias_output_points"],177-len(keys))

    def test_missing_control_qualification_blocks_all132_even_keep1(self):
        parts = fixtures(controls=True);parts[1][0]["roles"][-1]["control_qualification"] = "unreviewed"
        parts[1][0]["review"]["roles_sha256"] = formal.canonical_sha(parts[1][0]["roles"])
        p = planner(parts);rows = p.source_plan(1,p.inventory[0])["requests"]
        self.assertTrue(all(r["status"] == "eligible_proxy" for r in rows[:45]))
        self.assertEqual(sum(r["status"] == "blocked" for r in rows[45:]),132)
        self.assertTrue(all(r["positions"] is None for r in rows[45:] if r["keep_ratio"] == 1))

    def test_nested_masks_share_depth_order_and_report_discrete_retention(self):
        p = planner(fixtures(controls=True));rows = p.source_plan(1,p.inventory[0])["requests"]
        for region in (formal.PRIMARY,)+formal.CONTROLS:
            previous = None
            for keep in formal.KEEPS:
                same = [r for r in rows if r["region"] == region and r["keep_ratio"] == keep]
                self.assertEqual(len({tuple(r["positions"]) for r in same}),1)
                mask = set(same[0]["positions"])
                if previous is not None:self.assertTrue(mask <= previous)
                previous = mask
                for row in same:self.assertEqual(row["actual_retained_fraction"],(row["pool_token_count"]-len(mask))/row["pool_token_count"])
        a = next(r for r in rows if r["region"] == formal.PRIMARY and r["layer"] == 0 and r["keep_ratio"] == 0)
        b = next(r for r in rows if r["region"] == formal.PRIMARY and r["layer"] == 0 and r["keep_ratio"] == .1)
        c = next(r for r in rows if r["region"] == formal.PRIMARY and r["layer"] == 10 and r["keep_ratio"] == 0)
        self.assertEqual(a["success_cache_key"],b["success_cache_key"])
        self.assertNotEqual(a["success_cache_key"],c["success_cache_key"])
        self.assertTrue(a["cache_reuse_requires_prior_verified_success"])

    def test_area60_is_individual_strict_gt_never_a_union_limit(self):
        parts = fixtures();parts[1][0]["roles"][0]["xyxy_px"] = [0,0,100,61]
        parts[1][0]["review"]["roles_sha256"] = formal.canonical_sha(parts[1][0]["roles"])
        p = planner(parts);result = p.source_plan(1,p.inventory[0])
        self.assertEqual(len(result["rejected_role_boxes"]),1)
        self.assertTrue(all(r["status"] == "blocked" for r in result["requests"]))
        parts = fixtures();parts[1][0]["roles"][0]["xyxy_px"] = [0,0,100,60]
        parts[1][0]["roles"][1]["xyxy_px"] = [0,40,100,100]
        parts[1][0]["review"]["roles_sha256"] = formal.canonical_sha(parts[1][0]["roles"])
        p = planner(parts);result = p.source_plan(1,p.inventory[0])
        self.assertFalse(result["rejected_role_boxes"])
        self.assertTrue(result["models"]["openvla"]["primary_source_eligible"])
        self.assertEqual(result["models"]["openvla"]["primary_token_count"],16)

    def test_unreviewed_ambiguous_or_unbound_annotation_never_eligible(self):
        for key,value in (("status","detector_completed"),("ambiguous",True),("roles_sha256","0"*64),("audit_link","")):
            parts = fixtures();parts[1][0]["review"][key] = value
            p = planner(parts)
            self.assertTrue(all(r["status"] == "blocked" for r in p.source_plan(1,p.inventory[0])["requests"]))

    def test_actual_geometry_is_required_not_guessed_from_model_name(self):
        parts = fixtures();parts[2].clear();p = planner(parts)
        self.assertIn("actual_model_geometry_missing",p.source_plan(1,p.inventory[0])["models"]["openvla"]["primary_blocked_reasons"])
        for key,value in (("actual_geometry_verified",False),("model_identity_sha256","0"*64),("processor_sha256","0"*64)):
            parts = fixtures();parts[2][0][key] = value;p = planner(parts)
            self.assertFalse(p.source_plan(1,p.inventory[0])["models"]["openvla"]["primary_source_eligible"])

    def test_pi_qwen_gt_gate_false_and_measured_pi_state_required(self):
        parts = fixtures(models=("pi0","qwen_h16"));p = planner(parts);result = p.source_plan(1,p.inventory[0])
        self.assertEqual(len(result["requests"]),90)
        self.assertTrue(all(r["status"] == "eligible_proxy" and r["gt_mse_allowed"] is False for r in result["requests"]))
        altered = copy.deepcopy(parts);altered[3][0]["action_mapping"]["gt_mapping_verified"] = True
        altered[3][0]["identity"]["action_mapping_sha256"] = formal.canonical_sha(altered[3][0]["action_mapping"])
        with self.assertRaises(ValueError):planner(altered)
        next(g for g in parts[2] if g["model"] == "pi0").pop("state_input_audit")
        parts[1][0]["review"]["source_row_sha256"] = formal.canonical_sha(parts[0][0])
        for geometry in parts[2]:geometry["source_row_sha256"] = formal.canonical_sha(parts[0][0])
        p = planner(parts);result = p.source_plan(1,p.inventory[0])
        self.assertFalse(result["models"]["pi0"]["primary_source_eligible"])
        self.assertTrue(result["models"]["qwen_h16"]["primary_source_eligible"])

    def test_pi_native_state_proof_does_not_invent_frame_or_clip_sensor(self):
        parts = fixtures(models=("pi0",));source=parts[0][0];geometry=parts[2][0]
        self.assertTrue(formal.pi_measured_input_verified(source,geometry))
        for malformed in (None,[],{"measured_state7":None},{"measured_state7":[],"native_state8_before_normalization":None}):
            self.assertFalse(formal.pi_measured_input_verified(source,{"state_input_audit":malformed}))
        source.pop("state_frame")
        self.assertTrue(formal.pi_measured_input_verified(source,geometry))
        for key,value in [("source_array_sha256","0"*64),("original_anchor",7),("zero_state_substitute_used",True),
                          ("extra_axis_or_origin_transform_applied",True),("physical_frame_origin_verified_by_preprocessing",True)]:
            changed=copy.deepcopy(geometry);changed["state_input_audit"][key]=value
            self.assertFalse(formal.pi_measured_input_verified(source,changed))
        changed=copy.deepcopy(geometry);changed["state_input_audit"]["native_state8_before_normalization"][-1]=1.0
        self.assertFalse(formal.pi_measured_input_verified(source,changed))

    def test_goal_status_needs_bound_requirement_review_not_missing_box_default(self):
        parts=fixtures();source=parts[0][0];annotation=parts[1][0]
        for field,value in [("reason",""),("source_row_sha256","0"*64),("spatial_target_required",True),
                            ("policy_results_used",True)]:
            changed=copy.deepcopy(annotation);changed["goal_requirement_review"][field]=value
            reasons,_,_=formal._annotation(source,changed,"synthetic-bank-v1")
            self.assertTrue(any("goal_requirement" in r for r in reasons))
        changed=copy.deepcopy(annotation);changed.pop("goal_requirement_review")
        self.assertIn("goal_requirement_review_unbound",formal._annotation(source,changed,"synthetic-bank-v1")[0])
        changed=copy.deepcopy(annotation);changed["review"]["status"]="reviewed_blocked"
        reasons,_,_=formal._annotation(source,changed,"synthetic-bank-v1")
        self.assertIn("ai_review_not_qualified",reasons);self.assertNotIn("ai_review_not_completed",reasons)

    def test_qwen_reinjection_note_is_metadata_and_does_not_move_layer0(self):
        parts = fixtures(models=("qwen_h16",));parts[3][0]["protocol_notes"] = ["Synthetic fixture note, not real review"]
        p = planner(parts);metadata = p.manifest();note = metadata["model_protocol_notes"]["qwen_h16"]
        self.assertEqual(note["intervention"],"single_layer_input_intervention")
        self.assertTrue(note["deepstack_branch_unchanged"])
        self.assertEqual(note["native_deepstack_reinjection_after_layers"],[0,1,2])
        self.assertEqual(metadata["paper_layers"]["qwen_h16"],[0,12,24,34])
        layers = {r["layer"] for r in p.source_plan(1,p.inventory[0])["requests"] if r["region"] == formal.PRIMARY}
        self.assertEqual(layers,{0,12,24,34})

    def test_action_mapping_seal_duplicates_foreign_sources_and_seed_are_rejected(self):
        parts = fixtures();parts[3][0]["action_mapping"]["gt_mapping_verified"] = False
        with self.assertRaises(ValueError):planner(parts)
        parts = fixtures();parts[1].append(copy.deepcopy(parts[1][0]))
        with self.assertRaises(ValueError):planner(parts)
        parts = fixtures();parts[1][0]["sample_id"] = "foreign"
        with self.assertRaises(ValueError):planner(parts)
        parts = fixtures();parts[4]["seed"] = 7
        with self.assertRaises(ValueError):planner(parts)

    def test_partial_stream_cannot_publish_manifest_or_overwrite_existing(self):
        p = planner(fixtures());one = p.source_plan(1,p.inventory[0]);p.iter_sources = lambda:iter([one])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"new"
            with self.assertRaises(RuntimeError):formal.write_plan(p,output)
            self.assertFalse((output/"manifest.json").exists())
            before = (output/"requests.jsonl").read_bytes()
            self.assertEqual(len(before.splitlines()),45)
            with self.assertRaises(FileExistsError):formal.write_plan(p,output)
            self.assertEqual((output/"requests.jsonl").read_bytes(),before)


if __name__ == "__main__":unittest.main()
