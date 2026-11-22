"""Reusable diagnostics extracted from validated native implementations."""
import hashlib
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while (block := stream.read(8 * 1024 * 1024)):
            digest.update(block)
    return digest.hexdigest()

