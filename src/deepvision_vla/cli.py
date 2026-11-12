"""Discoverable command interface; imports no model, CUDA, or simulator."""
import argparse
import importlib
import sys

COMMANDS = {
    "image": ("evaluation.runner", "Predict from one image and explicit instruction"),
    "eval": ("evaluation.runner", "Evaluate one RLBench task on a selected GPU"),
    "infer": ("evaluation.runner", "Check a labeled dataset fixture (not required for image mode)"),
    "doctor": ("tools.doctor", "Check native libraries without model/GPU/simulator startup"),
    "configure": ("tools.configure", "Generate paths and a matching DINO runtime view"),
    "download": ("tools.download", "Download the pinned RLBench-10 checkpoint; --plan-only is offline"),
    "verify": ("tools.verify", "Verify the twenty checkpoint files by SHA256"),
    "prepare-rlbench": ("tools.prepare_rlbench", "Fetch the pinned native simulator dependencies"),
    "prepare-dino": ("tools.prepare_dino", "Build a view of the checkpoint's saved DINO component"),
    "audit-image": ("evaluation.audit_image", "Audit an existing image prediction without replay"),
    "audit-inference": ("evaluation.audit_inference", "Audit an existing dataset prediction without replay"),
    "audit-rollouts": ("evaluation.audit_rollouts", "Audit existing per-episode RLBench results"),
    "evaluate-all": ("evaluation.formal", "Explicitly start the fixed ten-task, 500-request protocol"),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="deepvision", description=__doc__)
    parser.add_argument("command", nargs="?", choices=COMMANDS, help="Operation to run")
    parser.epilog = "\n".join(f"  {name:18} {description}" for name, (_, description) in COMMANDS.items())
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return
    if arguments[0] not in COMMANDS:
        parser.error("unknown command: " + arguments[0])
    command, *remaining = arguments
    module = importlib.import_module("deepvision_vla." + COMMANDS[command][0])
    # Preserve the accepted runner's positional mode and all existing flags.
    forwarded = [command, *remaining] if command in ("image", "eval", "infer") else remaining
    previous = sys.argv
    try:
        sys.argv = ["deepvision " + command, *forwarded]
        module.main()
    finally:
        sys.argv = previous
