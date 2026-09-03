"""Progress reporting for the long loops: training and sampling.

Two things are wanted from a run that takes half an hour, and they are not the
same thing. A progress bar answers "how much longer", is redrawn in place, and
is worthless once the run is over. A periodic log answers "was it converging",
is one line per interval, and is the thing you read afterwards. Both are
provided here, and they are deliberately kept on separate streams: the bar
writes to stderr, the log to stdout, so piping a run to a file keeps the log
readable instead of interleaving thousands of carriage returns.

Set ``DGEOM_NO_PROGRESS=1`` to disable bars, which is what you want for a job
whose output is being captured rather than watched.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterable
from typing import Any

from tqdm.auto import tqdm

BAR_FORMAT = (
    "{desc} {percentage:3.0f}%|{bar:28}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


def bars_enabled() -> bool:
    """Whether progress bars should be drawn at all.

    Off when the environment asks, and off when stderr is not a terminal: a
    redrawn bar captured to a file is thousands of lines of carriage returns
    that bury the periodic log it was meant to accompany.
    """
    if os.environ.get("DGEOM_NO_PROGRESS", "") in ("1", "true", "yes"):
        return False
    return sys.stderr.isatty()


def track(
    iterable: Iterable,
    *,
    desc: str,
    total: int | None = None,
    leave: bool = False,
) -> Iterable:
    """Wrap an iterable in a progress bar.

    Args:
        iterable: what to iterate.
        desc: short label shown at the left of the bar.
        total: number of items, when the iterable does not report it.
        leave: keep the finished bar on screen.

    Returns:
        The iterable, wrapped when bars are enabled and bare otherwise.
    """
    if not bars_enabled():
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT,
        file=sys.stderr,
    )


def write(message: str) -> None:
    """Print a log line without corrupting an active progress bar."""
    if bars_enabled():
        tqdm.write(message, file=sys.stdout)
    else:
        print(message, flush=True)


# Fields that are fractions of one and belong on screen as percentages. The
# tables elsewhere print these with ".1%", and having the same quantity appear as
# "1.462" in a log line and "146.2%" in a table is a reliable way to misread a
# run as three orders of magnitude better than it is.
PERCENT_FIELDS = frozenset({"max_deviation", "noise_floor", "dev", "floor"})


class PeriodicLogger:
    """One log line every ``every`` iterations, plus a rate.

    Kept separate from the bar so that the record of a run survives being piped
    to a file, where a bar leaves nothing useful behind.
    """

    def __init__(self, tag: str, total: int, every: int) -> None:
        self.tag, self.total, self.every = tag, total, every
        self.t0 = time.time()

    def due(self, step: int) -> bool:
        """Whether this iteration should emit a line."""
        return self.every > 0 and (step % self.every == 0 or step == self.total - 1)

    def log(self, step: int, **fields: Any) -> None:
        """Emit one line for this iteration."""
        rate = (step + 1) / max(time.time() - self.t0, 1e-9)
        parts = []
        for k, v in fields.items():
            if isinstance(v, int | float) and k in PERCENT_FIELDS:
                parts.append(f"{k} {v:.1%}")
            elif isinstance(v, int | float):
                parts.append(f"{k} {v:.4g}")
            else:
                parts.append(f"{k} {v}")
        extra = "  ".join(parts)
        write(
            f"[{self.tag}] step {step + 1:>7,}/{self.total:,}  "
            f"{extra}{'  ' if extra else ''}{rate:.0f} it/s"
        )
