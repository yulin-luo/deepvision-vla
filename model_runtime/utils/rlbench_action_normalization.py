"""Explicit action target normalization, shared by training and dataset metrics."""
import numpy as np
LEGACY = 'quantile_linear_clipped_v1'
UNCLIPPED = 'quantile_linear_unclipped_v1'
def normalize_action(action, low, high, mask, contract=LEGACY):
    if contract not in (LEGACY, UNCLIPPED):
        raise ValueError('Unknown action normalization contract: ' + str(contract))
    action = np.asarray(action, dtype=np.float32)
    normalized = 2 * (action - low) / (high - low + 1e-8) - 1
    if contract == LEGACY:
        normalized = np.clip(normalized, -1, 1)
    return np.where(mask, normalized, action)
