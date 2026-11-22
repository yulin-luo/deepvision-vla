"""Bridge-orig target relabeling used by OpenVLA.

    Algorithm reference: OpenVLA (MIT), prismatic/vla/datasets/rlds/oxe/
    transforms.py and utils/data_utils.py. See licenses/OpenVLA_MIT_LICENSE.txt.
    This adapter does not claim that other policies share this action space.
"""

import math


def binarize_gripper(actions):
    values = [float(x) for x in actions]
    if not values or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
        raise ValueError("Bridge gripper values must be finite and in [0, 1]")
    carry = values[-1]
    output = [0.0] * len(values)
    for index in reversed(range(len(values))):
        if values[index] > 0.95:
            carry = 1.0
        elif values[index] < 0.05:
            carry = 0.0
        output[index] = carry
    return output


def bridge_orig_targets(states, actions):
    """Return original frame indices and reached-state delta + gripper targets.

    The first frame is removed and the final no-action frame is discarded,
    matching OpenVLA's bridge_orig transform. Motion is next minus current
    EEF state, rather than the recorded raw action's first six channels.
    """
    if len(states) != len(actions) or len(states) < 3:
        raise ValueError("Bridge trajectory needs matching states/actions and at least 3 frames")
    if any(len(x) < 6 for x in states) or any(len(x) != 7 for x in actions):
        raise ValueError("Expected EEF state with 6+ channels and raw 7D actions")
    states = [[float(x) for x in row[:6]] for row in states]
    if not all(math.isfinite(x) for row in states for x in row):
        raise ValueError("Non-finite Bridge EEF state")
    gripper = binarize_gripper([x[-1] for x in actions[1:]])
    return [{"original_frame_index": i,
             "action": [states[i+1][j]-states[i][j] for j in range(6)]+[gripper[i-1]]}
            for i in range(1, len(states)-1)]
