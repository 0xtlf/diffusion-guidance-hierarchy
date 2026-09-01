"""Training.

Trainer        the loop; everything periodic is a Callback
TrainConfig    optimiser and schedule settings
MetricTracker  running mean and spread between log intervals
"""

from .callbacks import Callback, Checkpointer, Evaluator, MetricLogger
from .metrics import MetricTracker
from .trainer import EMA, TrainConfig, Trainer

__all__ = [
    "EMA",
    "Callback",
    "Checkpointer",
    "Evaluator",
    "MetricLogger",
    "MetricTracker",
    "TrainConfig",
    "Trainer",
]
