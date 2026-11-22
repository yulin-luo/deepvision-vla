"""Reusable diagnostics extracted from validated native implementations."""
from deepvision_vla.diagnostics.formal_plan import is_sha
from pathlib import Path, PurePosixPath

SOURCES = 1500

def validate_linked_inventory(inventory, original, files, receipt, inventory_sha, manifest_sha):
    if receipt.get('status') != 'complete' or receipt.get('samples') != SOURCES or receipt.get('data_files_rehashed') != 2 * SOURCES or (receipt.get('inventory_sha256') != inventory_sha) or (receipt.get('source_manifest_sha256') != manifest_sha):
        raise ValueError('Actual completed fixed1500 staging receipt is required')
    if len(inventory) != SOURCES or len(original) != SOURCES:
        raise ValueError('Exactly1500 original identities must remain')
    ids = [row['sample_id'] for row in inventory]
    if len(set(ids)) != SOURCES or ids != [row['sample_id'] for row in original]:
        raise ValueError('Original identity coverage/order changed')
    by_path = {item['path']: item for item in files}
    if len(by_path) != len(files) or len(by_path) != 2 * SOURCES:
        raise ValueError('Expected3000 distinct staged payload paths')
    for ordinal, (row, source) in enumerate(zip(inventory, original), 1):
        if row.get('ordinal') != ordinal or row.get('original_source') != source or row.get('image_sha256') != source.get('image_sha256') or (row.get('image_size') != [source.get('width'), source.get('height')]) or (row.get('anchor') != source.get('anchor')) or (row.get('instruction') != source.get('instruction')):
            raise ValueError('Inventory link differs from the unchanged original manifest')
        for field, hash_key in (('image_path', 'image_sha256'), ('arrays_path', 'arrays_sha256')):
            item = by_path.get(source.get(field))
            if item is None or not is_sha(row.get(hash_key)) or item.get('sha256') != row[hash_key]:
                raise ValueError('Image/array SHA is not bound to the already verified staged inventory')
    return by_path

def safe_payload_path(root, relative):
    logical = PurePosixPath(relative)
    if logical.is_absolute() or '..' in logical.parts or '\\' in relative:
        raise ValueError('Unsafe payload path')
    root = Path(root).resolve()
    path = root.joinpath(*logical.parts).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Payload escapes the original staged root')
    return path

def failure_status(error):
    return 'missing_source' if isinstance(error, FileNotFoundError) else 'budget_unattempted' if isinstance(error, TimeoutError) else 'blocked_or_error'

