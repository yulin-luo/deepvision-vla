"""Prepare BridgeV2 inputs, execute a verified run, and redraw diagnostics."""

import argparse
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m deepvision_vla.diagnostics",
        description=__doc__,
        epilog="Prepare modes: fetch, data, plan, launch. Plot modes: redraw, aggregate, attribution, verify. Run validates a launch plan by default.",
    )
    parser.add_argument("command", choices=("prepare", "run", "plot"))
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command, *remainder = arguments
    if command == "prepare":
        mode_parser = argparse.ArgumentParser(prog=f"{parser.prog} prepare")
        mode_parser.add_argument("mode", choices=("fetch", "data", "plan", "launch"))
        if not remainder or remainder[0] in ("-h", "--help"):
            mode_parser.print_help()
            return 0
        mode, *remainder = remainder
        if mode == "fetch":
            from .fetch_bridge_shards import main as operation
        elif mode == "data":
            from .reconstruct_bridge import main as operation
        elif mode == "plan":
            from .formal_plan import main as operation
        elif mode == "launch":
            from .prepare_launch import main as operation
        else:
            mode_parser.error(f"unknown preparation mode: {mode}")
        subcommand = f"{command} {mode}"
    elif command == "run":
        from .formal_single_model_watchdog import main as operation
        subcommand = command
    elif command == "plot":
        mode_parser = argparse.ArgumentParser(prog=f"{parser.prog} plot")
        mode_parser.add_argument("mode", choices=("redraw", "aggregate", "attribution", "verify"))
        if not remainder or remainder[0] in ("-h", "--help"):
            mode_parser.print_help()
            return 0
        mode, *remainder = remainder
        if mode == "redraw":
            from .formal_reporting_plots import main as operation
        elif mode == "aggregate":
            from .formal_reporting import main as operation
        elif mode == "attribution":
            from .attribution_plot import main as operation
        elif mode == "verify":
            from .formal_reporting_verify import main as operation
        else:
            mode_parser.error(f"unknown plotting mode: {mode}")
        subcommand = f"{command} {mode}"
    else:
        parser.error(f"unknown command: {command}")

    # The component entry points also remain independently executable.
    previous = sys.argv
    try:
        sys.argv = [f"{parser.prog} {subcommand}", *remainder]
        return operation()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
