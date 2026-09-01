"""Name -> factory registries so YAML can select interchangeable components.

Each pluggable family owns a registry; a config names a component by string and
the registry resolves it. That is what lets a sampling scheme or a score model be
swapped in the config rather than in code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Maps a name to a class, so configs can select components by string."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Class decorator that records a class under `name`."""

        def deco(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._entries[name] = cls
            return cls

        return deco

    def build(self, name: str, **kwargs: Any) -> Any:
        """Construct a registered component by name."""
        if name not in self._entries:
            raise KeyError(
                f"unknown {self.kind} '{name}'. available: {sorted(self._entries)}"
            )
        return self._entries[name](**kwargs)

    def names(self) -> list[str]:
        """Sorted names of everything registered."""
        return sorted(self._entries)


SAMPLERS = Registry("sampler")
MODELS = Registry("diffusion model")
