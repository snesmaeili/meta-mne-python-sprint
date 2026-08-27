"""Epoch factories for the variable-duration browser sweep.

Every expectation returned here is computed from the source arrays, never from
the object under test, so a wrong ``_n_times_per_epoch`` or a wrong boundary
cannot make its own test pass.
"""

from dataclasses import dataclass, field

import numpy as np

from mne import EpochsArray, create_info

CH_CYCLE = ("eeg", "eeg", "eeg", "mag", "grad", "eeg", "stim", "misc")


@dataclass(frozen=True)
class Spec:
    """One point in the data matrix."""

    lengths: tuple
    n_channels: int = 3
    sfreq: float = 100.0
    tmin: str = "zero"  # zero | negative | mixed | positive
    drop: tuple = ()  # source indices dropped before plotting
    n_bads: int = 0
    mixed_types: bool = False
    force_fixed: bool = False  # build the 3-D fixed path even if lengths allow it
    seed: int = 0
    name: str = ""

    def label(self):
        return self.name or (
            f"n{len(self.lengths)}_ch{self.n_channels}_sf{self.sfreq:g}_{self.tmin}"
            + (f"_drop{len(self.drop)}" if self.drop else "")
            + ("_fixed" if self.force_fixed else "")
        )


@dataclass
class Case:
    """A built instance plus everything the invariants compare against."""

    epochs: object
    spec: Spec
    lengths: np.ndarray
    boundary_samples: np.ndarray
    boundary_times: np.ndarray
    source: list = field(default_factory=list)
    tmins: np.ndarray = None
    sfreq: float = 100.0

    @property
    def n_epochs(self):
        return len(self.lengths)


def _tmins(mode, n):
    if mode == "zero":
        return np.zeros(n)
    if mode == "negative":
        return np.full(n, -0.2)
    if mode == "positive":
        return np.full(n, 0.1)
    if mode == "mixed":
        return np.array([(0.0, -0.2, 0.1, -0.5)[i % 4] for i in range(n)])
    raise ValueError(f"unknown tmin mode {mode!r}")


def _channels(spec):
    if spec.mixed_types:
        types = [CH_CYCLE[i % len(CH_CYCLE)] for i in range(spec.n_channels)]
    else:
        types = ["eeg"] * spec.n_channels
    names = [f"{t.upper()}{i:03d}" for i, t in enumerate(types)]
    return names, types


def build(spec):
    """Return a :class:`Case` for one :class:`Spec`."""
    rng = np.random.default_rng(spec.seed)
    names, types = _channels(spec)
    info = create_info(names, spec.sfreq, types)
    if spec.n_bads:
        info["bads"] = names[: spec.n_bads]

    n_src = len(spec.lengths)
    arrays = [rng.standard_normal((len(names), n)) * 1e-6 for n in spec.lengths]
    tmins = _tmins(spec.tmin, n_src)

    # events far enough apart that no epoch's first sample goes negative
    stride = int(max(spec.lengths)) + int(spec.sfreq) + 100
    events = np.column_stack(
        [
            np.arange(n_src) * stride + stride,
            np.zeros(n_src, int),
            np.ones(n_src, int),
        ]
    )

    if spec.force_fixed:
        assert len(set(spec.lengths)) == 1, "force_fixed needs equal lengths"
        data = np.stack(arrays)
        epochs = EpochsArray(
            data,
            info,
            events=events,
            tmin=float(tmins[0]),
            event_id={"x": 1},
            baseline=None,
            verbose=False,
        )
    else:
        epochs = EpochsArray(
            arrays,
            info,
            events=events,
            tmin=tmins,
            event_id={"x": 1},
            baseline=None,
            verbose=False,
        )

    if spec.drop:
        epochs.drop(list(spec.drop), verbose=False)

    kept = [i for i in range(n_src) if i not in set(spec.drop)]
    lengths = np.array([spec.lengths[i] for i in kept], int)
    boundary_samples = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    return Case(
        epochs=epochs,
        spec=spec,
        lengths=lengths,
        boundary_samples=boundary_samples,
        boundary_times=boundary_samples / spec.sfreq,
        source=[arrays[i] for i in kept],
        tmins=tmins[kept],
        sfreq=spec.sfreq,
    )


class _AsFixedTripwire:
    """Raise if the browser ever falls back to padding."""

    def __call__(self, *args, **kwargs):
        raise AssertionError("as_fixed() was called: the browser padded")


def arm_tripwire(epochs):
    """Make any ``as_fixed()`` call a hard failure (invariant I6)."""
    epochs.as_fixed = _AsFixedTripwire()
    return epochs
