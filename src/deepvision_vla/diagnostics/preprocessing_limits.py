"""Reusable diagnostics extracted from validated native implementations."""
import os

CPU_THREADS = 2

def _cpu_limits():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        selected = sorted(os.sched_getaffinity(0))[:CPU_THREADS]
        if not selected:
            raise RuntimeError('No allowed CPU affinity')
        os.sched_setaffinity(0, selected)
        affinity = sorted(os.sched_getaffinity(0))
    return {'cpu_threads': 2, 'process_cpu_affinity': affinity, 'budget_seconds': 600, 'cuda_visible_devices': ''}

