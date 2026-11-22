"""Observed child memory with a bounded terminal-exit grace, never a zero fallback."""
import subprocess
import math


class MemoryObservationUnavailable(RuntimeError):
    pass


def observe_child_memory(child, status_text, *, remaining_seconds, exit_grace_seconds=1.0):
    """Return live RSS/HWM, or a known terminal return code.

    Linux can remove VmRSS/VmHWM while an exiting task has not yet become
    waitable. Wait once, bounded by the original deadline, before declaring
    a live observation failure. Unknown memory is retained as None.
    The caller owns PID/start-time checks and any resource-limit enforcement.
    """
    for value in (remaining_seconds,exit_grace_seconds):
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError("Finite numeric observation budgets are required")
    code = child.poll()
    if code is not None:
        return {'status':'terminal','returncode':code,'rss_bytes':None,'peak_rss_bytes':None}
    observed = {}
    for field,output in [('VmRSS','rss_bytes'),('VmHWM','peak_rss_bytes')]:
        values = [line.split(':',1)[1].strip().split() for line in (status_text or '').splitlines()
                  if line.startswith(field+':')]
        if (len(values)!=1 or len(values[0])!=2 or values[0][1]!='kB'
                or not values[0][0].isdigit()):
            grace = min(1.0,float(exit_grace_seconds),max(0.0,float(remaining_seconds)))
            if grace <= 0:
                raise TimeoutError('Original child observation deadline exhausted')
            try:
                code = child.wait(timeout=grace)
            except subprocess.TimeoutExpired as error:
                raise MemoryObservationUnavailable('Child remains live without a complete memory observation') from error
            return {'status':'terminal','returncode':code,'rss_bytes':None,'peak_rss_bytes':None,
                    'memory_fields_unavailable_during_exit':True,'exit_grace_budget_seconds':grace}
        observed[output] = int(values[0][0])*1024
    return {'status':'live','returncode':None,**observed}
