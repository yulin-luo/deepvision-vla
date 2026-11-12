"""Locations for the editable source distribution and its sealed runtime."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
ASSETS_ROOT = PACKAGE_ROOT / "assets"
MODEL_RUNTIME_ROOT = REPOSITORY_ROOT / "model_runtime"
LICENSES_ROOT = REPOSITORY_ROOT / "licenses"


def require_source_checkout():
    """Reject installations missing the scientific source snapshot."""
    evaluator = MODEL_RUNTIME_ROOT / "scripts_rlbench10_ablation" / "test_rlbench_no_gen_encoder.py"
    if not evaluator.is_file():
        raise FileNotFoundError(
            "This preview requires a source checkout: install with pip install -e . --no-deps. "
            "The model_runtime directory must remain next to pyproject.toml."
        )
    return MODEL_RUNTIME_ROOT
