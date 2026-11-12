"""Diagnose the configured native runtime without loading a model or simulator."""
import argparse
import json
from pathlib import Path
from deepvision_vla.runtime.native import diagnose_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).expanduser().read_text())
    if not isinstance(config, dict):
        parser.error('config must be a JSON object')
    report = diagnose_native(config)
    print(json.dumps(report, indent=2))
    if report['status'] not in ('ready', 'not_required'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
