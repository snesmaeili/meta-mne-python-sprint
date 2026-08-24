"""First-class provenance for temporal alignment.

mne-mobi could not do this: normalised gait phase had to be forced back through
``Epochs``' single ``sfreq``/``times`` model, so ``time_warp_epochs`` set
``info["sfreq"] = n_points - 1`` to fake a percent axis. After that point the
original durations were gone and "600 ms" and "60% of trial" were the same
number in the same field.

Here, alignment produces a record. The original durations survive it, and any
aligned object can say exactly how it was produced.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

__all__ = ["AlignmentRecord"]


@dataclasses.dataclass(frozen=True)
class AlignmentRecord:
    """What was done to get from native trial time to a common coordinate."""

    method: Literal[
        "none",
        "common-crop",
        "pad",
        "duration-normalise",
        "piecewise-linear",
    ]
    domain: Literal["signal", "tfr"]
    target_coord: Literal["seconds", "phase"]
    interpolation: str = "linear"

    #: per-epoch duration in seconds *before* alignment
    original_duration: np.ndarray | None = None
    #: per-epoch start/end in seconds relative to the recording
    original_start: np.ndarray | None = None
    original_end: np.ndarray | None = None
    #: per-epoch landmark latencies (s, relative to epoch t=0), ragged
    original_landmarks: list[np.ndarray] | None = None
    #: the common landmark latencies everything was mapped onto
    target_landmarks: np.ndarray | None = None
    #: how ``target_landmarks`` was chosen
    target_rule: str | None = None
    landmark_names: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.domain == "signal" and self.method == "piecewise-linear":
            # Not forbidden -- but the caller has to have chosen it knowingly.
            # See tests/test_frequency_preservation.py for why.
            pass

    @property
    def warps_spectral_content(self) -> bool:
        """True when this alignment shifts apparent frequency.

        Warping the signal before a time-frequency transform rescales the time
        axis the oscillations live on: a 10 Hz oscillation stretched by 2x
        reads as 5 Hz. Warping the TF representation instead moves energy along
        the time axis while leaving the frequency axis alone.
        """
        return self.domain == "signal" and self.method in (
            "duration-normalise",
            "piecewise-linear",
        )

    def summary(self) -> str:
        parts = [f"{self.method} in {self.domain} domain -> {self.target_coord}"]
        if self.target_rule:
            parts.append(f"target={self.target_rule}")
        if self.landmark_names:
            parts.append(f"landmarks={'/'.join(self.landmark_names)}")
        if self.warps_spectral_content:
            parts.append("WARNING: shifts apparent frequency")
        return "; ".join(parts)

    def __repr__(self) -> str:
        return f"<AlignmentRecord | {self.summary()}>"
