"""Per-task cooperative GPU leases on a local Linux filesystem.

This coordinates DeepVision launchers only. NVIDIA occupancy checks still
protect against unrelated applications which do not participate in this lease.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import stat
import time

_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


class GPULeaseUnavailable(RuntimeError):
    """The selected physical device already has a cooperating owner."""


def lease_path(gpu_uuid, lock_root=None):
    if not isinstance(gpu_uuid, str) or not _GPU_UUID.fullmatch(gpu_uuid):
        raise ValueError("a single physical NVIDIA GPU UUID is required")
    if lock_root is None:
        if os.name != "posix":
            raise RuntimeError("GPU execution requires Linux; lease tests may supply a local lock root")
        # Do not follow TMPDIR: an experiment may put it on an NFS output root.
        lock_root = Path("/tmp") / ("deepvision-v01-gpu-locks-" + str(os.getuid()))
    return Path(lock_root) / (gpu_uuid + ".lock")


@contextmanager
def gpu_lease(gpu_uuid, wait_seconds=0, lock_root=None):
    """Hold one UUID lease until the task and owned-session cleanup complete.

    The lock is never inherited by evaluator subprocesses or held by the batch
    controller. A competing ordinary or batch-launched task uses this same path.
    """
    if wait_seconds < 0:
        raise ValueError("wait_seconds must be non-negative")
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError("GPU leases require Linux/POSIX flock") from error
    path = lease_path(gpu_uuid, lock_root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise RuntimeError("refusing symlink GPU lease directory")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    parent_fd = os.open(path.parent, directory_flags)
    fd = None
    try:
        directory = os.fstat(parent_fd)
        if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.getuid():
            raise RuntimeError("GPU lease directory must be a directory owned by this user")
        if directory.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError("GPU lease directory is writable by another user/group")
        if path.is_symlink():
            raise RuntimeError("refusing symlink GPU lease file")
        # Open relative to the checked directory FD: renaming its path cannot
        # redirect this operation. Never chmod/unlink a pre-existing entry.
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        entry = os.fstat(fd)
        if not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.getuid() or entry.st_nlink != 1:
            raise RuntimeError("GPU lease must be a regular, singly linked file owned by this user")
        if entry.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError("GPU lease file is writable by another user/group")
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise
    finally:
        os.close(parent_fd)
    deadline = time.monotonic() + wait_seconds
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError as error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GPULeaseUnavailable("selected GPU is leased by another DeepVision task") from error
                time.sleep(min(0.1, remaining))
        yield {"gpu_uuid": gpu_uuid, "path": str(path), "owner_pid": os.getpid()}
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
