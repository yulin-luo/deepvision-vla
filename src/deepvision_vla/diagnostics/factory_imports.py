"""Reusable diagnostics extracted from validated native implementations."""
import os
from pathlib import Path
import sys
from deepvision_vla.diagnostics.provenance import sha256

def configure_factory_imports(model, factory):
    """Prepare genuine Pi source visibility before the unchanged factory import.

    Read pinned files only; never import torch, Transformers or LeRobot here.
    Reject an already-loaded different checkout instead of replacing modules.
    """
    if model != 'pi0':
        return {'status': 'not_required', 'model': model}
    from deepvision_vla.diagnostics.adapters.openpi_bridge import _DEPS
    source = Path(factory['lerobot_source_root']).resolve(strict=True)
    hashes = {}
    for relative, expected in _DEPS.items():
        path = source / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError('Pinned genuine LeRobot dependency differs: ' + relative)
        hashes[relative] = actual
    for name, module in tuple(sys.modules.items()):
        if name == 'lerobot' or name.startswith('lerobot.'):
            locations = [getattr(module, '__file__')] if getattr(module, '__file__', None) else list(getattr(module, '__path__', ()))
            if not locations or any((not Path(path).resolve().is_relative_to(source) for path in locations)):
                raise RuntimeError('A conflicting LeRobot module is already imported: ' + name)
    source_text = str(source)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    previous = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p and p != source_text]
    os.environ['PYTHONPATH'] = os.pathsep.join([source_text] + previous)
    return {'status': 'prepared_before_cpu_factory', 'model': model, 'lerobot_source_root': source_text, 'pythonpath_prepared': True, 'critical_source_sha256': hashes, 'neural_imports_performed': 0, 'modules_replaced': 0}

