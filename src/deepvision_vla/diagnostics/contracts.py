"""Explicit action spaces prevent invalid cross-model MSE comparisons."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ActionSpace:
    channels: tuple[str, ...]
    representation: str
    coordinate_frame: str
    translation_units: str
    rotation_units: str
    normalization: str
    definition_source: str
    verified: bool = False

    def __post_init__(self):
        if not self.channels or any(not isinstance(x, str) or not x.strip() for x in self.channels) or len(set(self.channels)) != len(self.channels):
            raise ValueError("Action channels must be unique and nonempty")
        if type(self.verified) is not bool:
            raise ValueError("Action verification must be a boolean")
        for field in (self.representation, self.coordinate_frame, self.translation_units,
                      self.rotation_units, self.normalization, self.definition_source):
            if not field.strip():
                raise ValueError("Every action-space field must be explicit")


def action_mse(prediction, reference, prediction_space, reference_space):
    if not prediction_space.verified or not reference_space.verified:
        raise ValueError("Ground-truth action mapping has not been verified")
    if prediction_space != reference_space:
        raise ValueError("Prediction and reference action definitions differ")
    expected = len(prediction_space.channels)
    if len(prediction) != expected or len(reference) != expected:
        raise ValueError("Action dimensions differ; truncation and padding are not allowed")
    values = [(float(a), float(b)) for a, b in zip(prediction, reference)]
    if not all(math.isfinite(a) and math.isfinite(b) for a, b in values):
        raise ValueError("Action values must be finite")
    return sum((a - b) ** 2 for a, b in values) / expected
