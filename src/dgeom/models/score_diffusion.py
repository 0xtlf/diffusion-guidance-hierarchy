"""A diffusion model backed by a score network."""

from __future__ import annotations

import torch

from ..nn import ScoreNetwork
from ..registry import MODELS
from .base import TrainableDiffusion, broadcast_sigma
from .schedule import NoiseSchedule


@MODELS.register("score")
class ScoreDiffusion(TrainableDiffusion):
    """Denoising score matching in hat space.

    With shat = sigma^2 * score and target x0 - x = -sigma z, the objective

        E || shat / sigma + z ||^2

    is ordinary epsilon-prediction written in hat coordinates.
    """

    def __init__(
        self,
        network: ScoreNetwork,
        schedule: NoiseSchedule,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(dim=network.dim, schedule=schedule)
        self.device = torch.device(device)
        # The network always runs in float32 regardless of the ambient default
        # dtype: geometry and metrics run in float64 on CPU, and MPS has no
        # float64 at all, so precision is a property of the model rather than of
        # whatever the caller last set globally.
        self.net = network.to(device=self.device, dtype=torch.float32)

    # ---------------------------------------------------------------- contract

    def shat(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """Evaluated with no grad and returned in the caller's dtype.

        Metrics and samplers run in float64 on CPU while the network runs in
        float32 on the accelerator, so the conversion lives here rather than
        being repeated at every call site.
        """
        sig = broadcast_sigma(sigma, x).squeeze(-1)
        with torch.no_grad():
            y = self.net(
                x.to(self.device, torch.float32), sig.to(self.device, torch.float32)
            )
        # cast only after leaving the accelerator: MPS cannot hold float64, so
        # y.to(device=..., dtype=float64) fails while y.cpu().double() is fine
        return y.detach().cpu().to(dtype=x.dtype).to(device=x.device)

    # ---------------------------------------------------------------- training

    def parameters(self):
        """Parameters to optimise."""
        return self.net.parameters()

    def train(self) -> None:
        """Switch to training mode."""
        self.net.train()

    def eval(self) -> None:
        """Switch to evaluation mode."""
        self.net.eval()

    def loss(self, x0: torch.Tensor, *, generator=None) -> torch.Tensor:
        """Scalar training loss for a batch of clean samples."""
        n = x0.shape[0]
        sigma = self.schedule.sample(n, generator=generator)
        noise = torch.randn(x0.shape, generator=generator)
        x = x0.to(torch.float32) + sigma.unsqueeze(-1) * noise
        pred = self.net(x.to(self.device), sigma.to(self.device))
        target = noise.to(self.device)
        return (
            ((pred / sigma.to(self.device).unsqueeze(-1) + target) ** 2).sum(-1).mean()
        )

    # ------------------------------------------------------------ persistence

    def state_dict(self) -> dict:
        """Weights, for checkpointing."""
        return self.net.state_dict()

    def load_state_dict(
        self, state: dict, strict: bool = True, freeze: bool = True
    ) -> None:
        """Strict by default, deliberately.

        A non-strict load silently accepts a total key mismatch and leaves the
        network at its random initialisation, which then looks like a model that
        trained badly rather than one that never loaded. That failure mode cost
        an entire debugging cycle; loading now raises instead.
        """
        self.net.load_state_dict(state, strict=strict)
        if freeze:
            self.net.eval()
            for param in self.net.parameters():
                param.requires_grad_(False)

    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        return self.net.n_params
