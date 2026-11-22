"""Supervise one native paper attribution probe with a bound Linux parent."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from deepvision_vla.diagnostics import formal_single_model_watchdog as guard
from deepvision_vla.diagnostics.formal_execution_core import DurableJSONL
from deepvision_vla.diagnostics.formal_plan import LAYERS
from deepvision_vla.diagnostics.paper_probe import process_identity, validate_configuration
from deepvision_vla.diagnostics.provenance import sha256 as file_sha256
from deepvision_vla.diagnostics.source_manifest import verify_source_manifest
from deepvision_vla.runtime.gpu_lease import lease_path


CLOCK_FIELDS = ("budget_origin_epoch", "budget_origin_monotonic", "deadline_epoch", "deadline_monotonic")
PARENT_FIELDS = ("parent", "parent_intent", "parent_intent_sha256")
MODULE = "deepvision_vla.diagnostics.paper_probe"
SOURCE_FIELDS = ("sample_id", "image_sha256", "arrays_sha256", "instruction", "anchor")


def read_bound(path, expected):
    with path.open("rb") as stream:
        raw = stream.read(20_000_001)
    if len(raw) > 20_000_000 or sha256(raw).hexdigest() != expected:
        raise ValueError(f"Metadata digest differs for {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a metadata object in {path}")
    return value


def create_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def bind_live_permit(template, output, source):
    parent = process_identity(os.getpid())
    command = Path("/proc/self/cmdline").read_bytes()
    source = source.resolve(strict=True)
    if os.fsencode(source) not in command.split(b"\0"):
        raise RuntimeError("Invoke the watchdog using its absolute script path")
    intent = {
        "schema": "deepvision_probe_parent_intent_v1",
        "parent": parent,
        "config_sha256": template["config_sha256"],
        "watchdog_source": str(source),
        "watchdog_source_sha256": file_sha256(source),
        "parent_cmdline_sha256": sha256(command).hexdigest(),
        "maximum_child_rss_bytes": template["maximum_rss_bytes"],
        "owned_process_group_only": True,
        "gpu_leases_owned_by_parent": False,
        **{key: template[key] for key in CLOCK_FIELDS},
    }
    intent_path = output / "parent_intent.json"
    create_json(intent_path, intent)
    permit_path = output / "permit.json"
    create_json(permit_path, {**template, "parent": parent,
        "parent_intent": str(intent_path), "parent_intent_sha256": file_sha256(intent_path)})
    return permit_path, parent


def preflight(plan_path, plan_sha):
    plan = read_bound(plan_path, plan_sha)
    if plan.get("schema") != "deepvision_paper_probe_launch_v1":
        raise ValueError("Expected a single-probe launch plan")
    for name in ("config", "permit_template", "model_python", "nvidia_smi"):
        guard.require_posix(plan.get(name))
    config = read_bound(Path(plan["config"]), plan["config_sha256"])
    template = read_bound(Path(plan["permit_template"]), plan["permit_template_sha256"])
    if any(template.get(name) is not None for name in PARENT_FIELDS):
        raise ValueError("The watchdog binds the actual live parent")
    source = Path(__file__).resolve(strict=True)
    binaries = (("model_python", Path(plan["model_python"])), ("nvidia_smi", Path(plan["nvidia_smi"])))
    for name, path in binaries:
        if not os.access(path, os.X_OK) or file_sha256(path) != plan[name+"_sha256"]:
            raise ValueError(f"Executable seal differs for {name}")
    if Path(plan["nvidia_smi"]).name != "nvidia-smi" or file_sha256(source) != plan["watchdog_sha256"]:
        raise ValueError("Expected exact watchdog and nvidia-smi executables")
    root = Path(guard.require_posix(config.get("package_source_root"))).resolve(strict=True)
    if source.parents[3] != root:
        raise ValueError("Probe configuration must bind the checkout containing this watchdog")
    manifest = read_bound(Path(config["source_manifest"]), config["source_manifest_sha256"])
    verify_source_manifest(root, manifest)
    paths = {record["path"] for record in manifest["files"]}
    if not {source.relative_to(root).as_posix(), Path(guard.__file__).resolve().relative_to(root).as_posix(),
            "src/deepvision_vla/diagnostics/paper_probe.py"}.issubset(paths):
        raise ValueError("The source manifest must include both watchdog modules and the real probe worker")
    out = Path(guard.require_posix(plan.get("parent_output"))).resolve()
    child_out = Path(guard.require_posix(config.get("output"))).resolve()
    child_once = Path(guard.require_posix(config.get("once"))).resolve()
    if any(path.exists() for path in (out, child_out, child_once)):
        raise FileExistsError("Existing scope output/once files are preserved")
    if out in child_out.parents or child_out in out.parents or out in child_once.parents:
        raise ValueError("Parent journals and probe worker outputs must be separate")
    model = validate_configuration(config, template, plan["config_sha256"],
        now_epoch=time.time(), now_monotonic=time.monotonic())
    sample = read_bound(Path(config["sample"]), config["sample_sha256"])
    return plan, config, template, model, root, out, {name: sample[name] for name in SOURCE_FIELDS}


def raw_map_complete(value):
    if not isinstance(value, dict) or value.get("display_normalization_applied") is not False:
        return False
    positions = value.get("visual_positions")
    if (not isinstance(positions, list) or not positions
            or any(type(pos) is not int or pos < 0 for pos in positions)
            or len(set(positions)) != len(positions)):
        return False
    for name in ("raw_signed", "raw_relu", "channel_weights"):
        numbers = value.get(name)
        if (not isinstance(numbers, list) or not numbers
                or any(type(number) not in (int, float) or not math.isfinite(number) for number in numbers)):
            return False
    return (len(value["raw_signed"]) == len(value["raw_relu"]) == len(positions)
            and all(relu == max(0, signed) for relu,signed in zip(value["raw_relu"],value["raw_signed"],strict=True)))


def terminal_matches(value, *, config_sha, permit_sha, model, sample):
    if not isinstance(value, dict):
        return False
    layers = LAYERS[model]
    probes = value.get("probes")
    if (value.get("status") != "complete" or value.get("model") != model
            or value.get("config_sha256") != config_sha or value.get("permit_sha256") != permit_sha
            or value.get("lease_released") is not True or value.get("model_loads") != 1
            or value.get("probes_completed") != len(layers) or value.get("backward_calls") != len(layers)
            or any(value.get(name) != sample[name] for name in SOURCE_FIELDS)
            or not isinstance(probes, list) or len(probes) != len(layers)):
        return False
    key = "logical_output_layer" if model == "pi0" else "layer"
    return all(isinstance(row, dict) and row.get(key) == layer and row.get("status") == "complete"
               and raw_map_complete(row.get("contribution"))
               and type(row.get("scalar_value")) in (int, float) and math.isfinite(row["scalar_value"])
               for row, layer in zip(probes, layers, strict=True))


def run(plan_path, plan_sha, *, execute=False):
    plan, config, template, model, root, out, sample = preflight(plan_path, plan_sha)
    if not execute:
        return {"status": "launch_metadata_validated", "model": model, "child_attempts": 0}
    if plan.get("launch_authorized") is not True:
        raise ValueError("The launch plan did not authorize execution")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Run the watchdog with Linux Python supporting pidfds")
    fd = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(fd, 0)
    finally:
        os.close(fd)
    source = Path(__file__).resolve(strict=True)
    if os.fsencode(source) not in Path("/proc/self/cmdline").read_bytes().split(b"\0"):
        raise RuntimeError("Invoke the watchdog using its absolute script path")
    guard.local_filesystem(out)
    guard.assert_no_parent_gpu_lock_fds()
    guard.enable_subreaper()
    out.mkdir(parents=True, exist_ok=False)
    permit, parent = bind_live_permit(template, out, source)
    permit_sha = file_sha256(permit)
    helper = out / "watchdog_json_reader.py"
    with helper.open("xb") as stream:
        stream.write(Path(guard.__file__).read_bytes())
    helper.chmod(0o444)
    probes = {"python": sys.executable, "script": str(helper),
              "nvidia_smi": plan["nvidia_smi"], "cwd": str(out)}
    command = [plan["model_python"], "-u", "-m", MODULE, "--config", plan["config"],
               "--config-sha256", plan["config_sha256"], "--permit", str(permit),
               "--permit-sha256", permit_sha, "--execute"]
    tmp = out / "tmp"
    tmp.mkdir()
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1",
        PATH=str(Path(plan["nvidia_smi"]).parent)+os.pathsep+os.environ["PATH"],
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
    result = {"status": "starting", "model": model, "launch_plan_sha256": plan_sha,
        "parent": parent, "child_attempts": 0, "permit_sha256": permit_sha, "command": command}
    create_json(out / "launch_intent.json", result)
    events = DurableJSONL(out / "parent_events.jsonl")
    child = None
    child_fd = None
    family = None
    monitor = None

    def save_event(event):
        event = dict(event)
        name = event.pop("event")
        events.append(name, **event)

    def interrupted(_number, _frame):
        raise InterruptedError("Attribution watchdog interrupted")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        validate_configuration(config, template, plan["config_sha256"],
            now_epoch=time.time(), now_monotonic=time.monotonic())
        with (out / "child_stdout.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=out, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            result["child_attempts"] = 1
            child_fd = os.pidfd_open(child.pid)
            identity = guard.proc_identity(child.pid)
            if identity is None:
                raise RuntimeError("Private child identity unavailable")
            family = guard.OwnedFamily(identity)
            save_event({"event": "child_started", "worker": identity})
            _, monitor = guard.monitor_private_child(child, identity,
                deadlines={"absolute_deadline_epoch": template["deadline_monotonic"]},
                cap=template["maximum_rss_bytes"], natural_grace=1, term_grace=5,
                kill_grace=3, poll_seconds=.25, family=family, on_event=save_event)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        if family is not None and monitor is None:
            _, monitor = guard.monitor_private_child(child, family.root,
                deadlines={"absolute_deadline_epoch": time.monotonic()},
                cap=template["maximum_rss_bytes"], natural_grace=0, term_grace=5,
                kill_grace=3, poll_seconds=.25, family=family, on_event=save_event)
        elif child is not None and family is None:
            # Reap only the unreaped direct child when /proc cannot establish its family.
            try:
                if child_fd is None:
                    child.kill()
                else:
                    signal.pidfd_send_signal(child_fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=3)
    finally:
        if child_fd is not None:
            os.close(child_fd)
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            terminal = {"status": "unknown", "reason": "no_verified_child"}
            release = {"owned_resource_release_verified": False}
            if family is not None:
                uuid = template["expected_gpu_uuid"]
                gpu = guard.observe_selected_nvml(uuid, probes)
                lock = guard.kernel_lock_snapshot(lease_path(uuid))
                snapshot, unreadable = guard.proc_snapshot()
                release = guard.release_assessment(family, snapshot, unreadable, gpu, lock)
                terminal = guard.isolated_json_read(Path(config["output"]) / "result.json", 3, probes)
            verified = terminal.get("status") == "observed" and terminal_matches(
                terminal["value"], config_sha=plan["config_sha256"], permit_sha=permit_sha, model=model, sample=sample)
            stop = (monitor or {}).get("stop_reason") or ("parent_exception" if "error" in result else None)
            status, code = guard.final_parent_status(child.returncode if child else None, verified, release, stop)
            if code == 0:
                status = "complete_attribution_independently_released"
            result.update(status=status, parent_exitcode=code, monitor=monitor,
                          independent_release=release, child_terminal_probe=terminal)
            guard.assert_no_parent_gpu_lock_fds()
            create_json(out / "result.json", result)
        finally:
            events.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--launch-plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.launch_plan, args.launch_plan_sha256, execute=args.execute)
    print(json.dumps({"status": result["status"], "model": result["model"],
                      "child_attempts": result["child_attempts"]}))
    return result.get("parent_exitcode", 0)


if __name__ == "__main__":
    raise SystemExit(main())
