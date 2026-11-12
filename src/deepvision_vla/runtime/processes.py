"""Bounded cleanup of a subprocess session created by this launcher only."""
import os
from pathlib import Path
import signal
import time


class ProcessCleanupError(RuntimeError):
    """A known owned session could not be confirmed stopped."""


def _stat(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "pgid": int(fields[2]),
                "session": int(fields[3]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def session_identity(proc):
    value = _stat(proc.pid)
    return value or {"pid": proc.pid, "state": None, "pgid": proc.pid,
                     "session": proc.pid, "start_ticks": None}


def _members(pgid):
    members = []
    try:
        entries = Path("/proc").iterdir()
        for entry in entries:
            if entry.name.isdigit():
                row = _stat(entry.name)
                if row and row["pgid"] == pgid and row["state"] != "Z":
                    members.append(row)
    except OSError as error:
        raise ProcessCleanupError("cannot inspect the owned Linux session") from error
    return members


def _token_matches(pid, token):
    try:
        values = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        return ("DEEPVISION_RUN_TOKEN=" + token).encode() in values
    except OSError:
        return False


def cleanup_owned_session(proc, identity, token, terminate_timeout=10, kill_timeout=5):
    """Stop the known new session, including children left after its leader exits.

    A live unreaped Popen child cannot have its PID reused. Once its leader has
    exited, we additionally require every remaining group member to carry this
    invocation's token and session ID before signalling. Unrelated processes are
    never selected by executable name, GPU number, username, or ancestor guesses.
    """
    if proc is None:
        return {"status": "no_child", "signals": []}
    if os.name != "posix":
        # The public GPU runtime is Linux-only. This direct-child fallback lets
        # CPU fault-injection tests run safely on Windows without group guessing.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=terminate_timeout)
            except __import__("subprocess").TimeoutExpired:
                proc.kill()
                proc.wait(timeout=kill_timeout)
        return {"status": "released", "signals": [], "scope": "direct_child_non_posix"}
    pgid = proc.pid
    if identity.get("pid") != pgid or identity.get("pgid") != pgid:
        raise ProcessCleanupError("refusing cleanup: child was not created as a new session")
    sent = []

    def live_owned():
        running = proc.poll() is None
        members = _members(pgid)
        if not members:
            return []
        if running:
            if os.getpgid(proc.pid) != pgid or os.getsid(proc.pid) != pgid:
                raise ProcessCleanupError("refusing cleanup: owned process session changed")
            observed = _stat(proc.pid)
            expected_ticks = identity.get("start_ticks")
            if observed and expected_ticks is not None and observed["start_ticks"] != expected_ticks:
                raise ProcessCleanupError("refusing cleanup: child identity changed")
        else:
            # A reused session leader would no longer have our child identity.
            observed = _stat(pgid)
            if observed and observed["start_ticks"] != identity.get("start_ticks"):
                raise ProcessCleanupError("refusing cleanup: process-group ID was reused")
            if any(row["session"] != pgid or not _token_matches(row["pid"], token)
                   for row in members):
                raise ProcessCleanupError("refusing cleanup: remaining process ownership is unproven")
        return members

    def signal_owned(signum):
        if not live_owned():
            return
        try:
            os.killpg(pgid, signum)
            sent.append(int(signum))
        except ProcessLookupError:
            pass

    def wait_stopped(timeout):
        deadline = time.monotonic() + timeout
        while True:
            if not live_owned():
                proc.poll()  # Reap the direct child if it exited between checks.
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    signal_owned(signal.SIGTERM)
    if not wait_stopped(terminate_timeout):
        signal_owned(signal.SIGKILL)
        if not wait_stopped(kill_timeout):
            raise ProcessCleanupError("owned session still live after bounded cleanup")
    return {"status": "released", "signals": sent, "process_group": pgid}

