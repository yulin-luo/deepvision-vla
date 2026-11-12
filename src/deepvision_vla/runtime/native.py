"""CPU-only native-library diagnosis before any model or simulator starts."""
import os
from pathlib import Path
import platform
import re
import subprocess


class NativeDependencyError(RuntimeError):
    """The configured native runtime is incomplete or cannot be inspected."""


def native_environment(config):
    if config.get('image_only') is True:
        return {}
    sim = Path(config['coppeliasim_root']).expanduser().resolve(strict=True)
    extras = config.get('native_library_paths', [])
    if not isinstance(extras, list) or not all(isinstance(p, str) and p for p in extras):
        raise NativeDependencyError('native_library_paths must be a list of directory paths')
    directories = [sim, *[Path(p).expanduser().resolve(strict=True) for p in extras]]
    if config.get('host_library_path'):
        directories.append(Path(config['host_library_path']).expanduser().resolve(strict=True))
    if not all(p.is_dir() for p in directories):
        raise NativeDependencyError('Native library search paths must be directories')
    paths = list(dict.fromkeys(str(p) for p in directories))
    return {'COPPELIASIM_ROOT': str(sim), 'LD_LIBRARY_PATH': os.pathsep.join(paths),
            'QT_QPA_PLATFORM_PLUGIN_PATH': str(sim / 'platforms'),
            'QT_PLUGIN_PATH': str(sim), 'QT_QPA_PLATFORM': 'xcb'}


def diagnose_native(config):
    report = {'status': 'missing', 'image_only': config.get('image_only', False),
              'missing_files': [], 'missing_libraries': [], 'errors': [], 'checks': [],
              'model_calls': 0, 'simulator_calls': 0, 'gpu_calls': 0}
    if config.get('image_only') is True:
        report['status'] = 'not_required'
        return report
    if platform.system() != 'Linux':
        report['status'] = 'unsupported'
        report['errors'].append('Native RLBench diagnostics require Linux and ldd')
        return report
    try:
        patch = native_environment(config)
    except (KeyError, OSError, ValueError, NativeDependencyError) as error:
        report['errors'].append(str(error))
        return report
    report['environment'] = patch
    env = dict(os.environ, **patch)
    sim = Path(config['coppeliasim_root']).expanduser().resolve()
    libraries = [sim / 'platforms/libqxcb.so', sim / 'libQt5Core.so.5',
                 sim / 'libQt5Gui.so.5', sim / 'libQt5Widgets.so.5']
    for library in libraries:
        if not library.is_file():
            report['missing_files'].append(str(library))
            continue
        try:
            result = subprocess.run(['ldd', str(library)], env=env, text=True,
                                    capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            report['errors'].append('ldd failed for ' + str(library) + ': ' + str(error))
            continue
        missing = re.findall(r'^\s*(\S+)\s+=>\s+not found\s*$', result.stdout, re.MULTILINE)
        report['missing_libraries'].extend(missing)
        report['checks'].append({'path': str(library), 'returncode': result.returncode,
                                 'missing': missing, 'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            report['errors'].append('ldd returned nonzero for ' + str(library))
    report['missing_libraries'] = sorted(set(report['missing_libraries']))
    if not report['errors'] and not report['missing_files'] and not report['missing_libraries']:
        report['status'] = 'ready'
    report['remediation'] = ('Install the exact conda native dependency lock, or point configure '
                             '--native-library-path at verified overlay lib directories. '
                             'Run deepvision doctor --config CONFIG before evaluation.')
    return report


def require_native_dependencies(config):
    report = diagnose_native(config)
    if report['status'] not in ('ready', 'not_required'):
        detail = ', '.join(report['missing_libraries'] + report['missing_files'] + report['errors'])
        raise NativeDependencyError('Native dependency check failed before model/GPU entry: '
                                    + detail + '. ' + report.get('remediation', 'Use Linux with ldd.'))
    return report
