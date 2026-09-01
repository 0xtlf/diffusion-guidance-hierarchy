"""Feature trunks. A backbone maps a conditioned input to features, no task head."""

from .mlp import MLPBackbone

__all__ = ["MLPBackbone"]
