"""Reusable diagnostics extracted from validated native implementations."""
from pathlib import Path

MIN_AVAILABLE = 24000000000

MAX_RSS = 32000000000

def parse_kb(text, key):
    values = [line.split(':', 1)[1].strip().split() for line in text.splitlines() if line.startswith(key + ':')]
    if len(values) != 1 or len(values[0]) != 2 or values[0][1] != 'kB':
        raise RuntimeError('Actual Linux memory field is missing/ambiguous: ' + key)
    return int(values[0][0]) * 1024

def memory_observation(meminfo, status, *, require_available=False):
    available = parse_kb(meminfo, 'MemAvailable')
    rss = parse_kb(status, 'VmRSS')
    peak = parse_kb(status, 'VmHWM')
    if rss > MAX_RSS or peak > MAX_RSS:
        raise MemoryError('Observed process RSS/HWM exceeds32GB bound')
    if require_available and available < MIN_AVAILABLE:
        raise MemoryError('Need at least24GB actual host memory before CPU policy load')
    return {'host_mem_available_bytes': available, 'rss_bytes': rss, 'peak_rss_bytes': peak, 'rss_limit_bytes': MAX_RSS, 'asynchronous_enforcement': 'external watchdog required during native materialization; observed RSS/HWM checked at stage boundaries and each source'}

def actual_memory(require_available=False):
    return memory_observation(Path('/proc/meminfo').read_text(encoding='utf-8'), Path('/proc/self/status').read_text(encoding='utf-8'), require_available=require_available)

