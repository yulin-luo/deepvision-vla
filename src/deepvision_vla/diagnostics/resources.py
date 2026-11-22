"""Explicit resource policy for a bounded diagnostic worker."""

import re
import ctypes


def nvidia_memory_bytes(value):
    match = re.fullmatch(r"\s*(\d+)\s+MiB\s*", str(value))
    if not match:
        raise ValueError("NVIDIA did not report a usable memory quantity")
    return int(match.group(1)) * 1024 * 1024


def pci_identity(value):
    domain, bus, device = value.strip().lower().split(":")
    return int(domain, 16), int(bus, 16), device


def verify_cuda_pci(expected_pci):
    """Check driver-visible device 0 after numeric selection, before Torch."""
    driver = ctypes.CDLL("libcuda.so.1")
    result = driver.cuInit(0)
    if result != 0:
        raise RuntimeError("CUDA driver initialization failed: " + str(result))
    count = ctypes.c_int()
    if driver.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value != 1:
        raise RuntimeError("Exactly one selected CUDA device is required")
    device = ctypes.c_int()
    bus = ctypes.create_string_buffer(32)
    if driver.cuDeviceGet(ctypes.byref(device), 0) != 0 or driver.cuDeviceGetPCIBusId(bus, 32, device) != 0:
        raise RuntimeError("CUDA device PCI address is unavailable")
    actual_pci = bus.value.decode()
    if pci_identity(actual_pci) != pci_identity(expected_pci):
        raise RuntimeError("CUDA device differs from the selected NVML PCI device")
    return {"driver_device_count": count.value, "actual_pci": actual_pci,
            "expected_pci": expected_pci}


def validate_device(info, *, own_pid, expected_uuid=None, allow_sharing=False,
                    minimum_free_bytes=0, maximum_own_bytes=None):
    if expected_uuid is not None and info["uuid"] != expected_uuid:
        raise RuntimeError("Selected GPU identity changed")
    foreign = [process for process in info["processes"] if process["pid"] != own_pid]
    if foreign and not allow_sharing:
        raise RuntimeError("Selected GPU has another process; sharing was not enabled")
    if minimum_free_bytes < 0:
        raise ValueError("Minimum free memory cannot be negative")
    if info["free_bytes"] < minimum_free_bytes:
        raise RuntimeError("Insufficient free memory for the declared diagnostic budget")
    if maximum_own_bytes is not None:
        own = [p.get("used_bytes") for p in info["processes"] if p["pid"] == own_pid]
        if any(value is not None and value > maximum_own_bytes for value in own):
            raise RuntimeError("Diagnostic process exceeded its observed GPU memory budget")
    return info
import subprocess
import xml.etree.ElementTree as ET
import os

def device_identity(index):
    raw = subprocess.check_output(['nvidia-smi', '-q', '-x', '-i', str(index)], text=True, timeout=15)
    devices = ET.fromstring(raw).findall('gpu')
    if len(devices) != 1:
        raise RuntimeError('Expected one selected physical GPU')
    device = devices[0]
    return {'index': index, 'uuid': device.findtext('uuid'), 'pci': device.findtext('pci/pci_bus_id'), 'free_bytes': nvidia_memory_bytes(device.findtext('fb_memory_usage/free')), 'total_bytes': nvidia_memory_bytes(device.findtext('fb_memory_usage/total')), 'processes': [{'pid': int(p.findtext('pid')), 'type': p.findtext('type'), 'used_bytes': nvidia_memory_bytes(p.findtext('used_memory')) if p.findtext('used_memory', '').strip().endswith('MiB') else None} for p in device.findall('processes/process_info')]}

def require_available(index, uuid=None, *, sharing=False, minimum_free_bytes=0, maximum_own_bytes=None):
    info = device_identity(index)
    return validate_device(info, own_pid=os.getpid(), expected_uuid=uuid, allow_sharing=sharing, minimum_free_bytes=minimum_free_bytes, maximum_own_bytes=maximum_own_bytes)

def require_own_memory(info, pid):
    own = [row for row in info['processes'] if row['pid'] == pid]
    if not own or any((row.get('used_bytes') is None for row in own)):
        raise RuntimeError('Own GPU process memory is unknown; budget cannot be verified')
    return max((row['used_bytes'] for row in own))

