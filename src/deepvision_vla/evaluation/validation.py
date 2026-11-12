"""Validation gates that remain active under Python optimization."""


def require(condition, message):
    if not condition:
        raise ValueError("invalid saved evidence: " + message)
