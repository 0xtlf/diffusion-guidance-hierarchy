"""Neural network components, organised by role.

    layers/     reusable pieces (embeddings, residual blocks)
    backbones/  feature trunks
    networks/   complete networks with a task-specific head

A new network type (a classifier, say) adds one file under networks/ and reuses
the existing layers and backbones.
"""

from .backbones import MLPBackbone
from .layers import FourierSigmaEmbedding, ResidualBlock
from .networks import ScoreNetwork

__all__ = ["FourierSigmaEmbedding", "MLPBackbone", "ResidualBlock", "ScoreNetwork"]
