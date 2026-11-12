"""Verify all twenty pinned files once and write reusable integrity evidence."""
import argparse
import json
from deepvision_vla.runtime.checkpoint import verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    result = verify_checkpoint(
        args.checkpoint, args.receipt,
        progress=lambda name, ok: print(name, 'PASS' if ok else 'FAIL', flush=True),
    )
    print(json.dumps({'status': result['status'], 'receipt': args.receipt,
                      'files': len(result['files']), 'policy': result['policy']}))
    if result['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
