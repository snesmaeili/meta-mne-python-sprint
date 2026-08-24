"""Storage backends for ragged epoch data.

The card that motivates this work names AwkwardArray. This module deliberately
puts three candidates behind one interface so the choice can be settled by
measurement (see ``benchmarks/container_backends.py``) rather than assumption.

Every backend stores ``n_epochs`` blocks of shape ``(n_channels, n_i)`` where
``n_i`` varies per epoch and ``n_channels`` is constant. That invariant --
*only time is ragged, never channels* -- is the whole point, and each backend
must enforce it structurally or by check.
"""

from __future__ import annotations

import abc

import numpy as np

__all__ = [
    "RaggedStore",
    "ListStore",
    "PaddedStore",
    "AwkwardStore",
    "get_backend",
    "BACKENDS",
]


class RaggedStore(abc.ABC):
    """Abstract per-epoch storage of ``(n_channels, n_i)`` blocks."""

    name: str = "abstract"

    # -- construction ---------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def from_list(cls, blocks: list[np.ndarray]) -> RaggedStore:
        """Build from a list of ``(n_channels, n_i)`` arrays."""

    # -- introspection --------------------------------------------------
    @property
    @abc.abstractmethod
    def lengths(self) -> np.ndarray:
        """``(n_epochs,)`` int array of per-epoch sample counts."""

    @property
    @abc.abstractmethod
    def n_channels(self) -> int: ...

    @property
    @abc.abstractmethod
    def nbytes(self) -> int:
        """Bytes held by the payload, excluding Python object overhead."""

    def __len__(self) -> int:
        return len(self.lengths)

    # -- access ---------------------------------------------------------
    @abc.abstractmethod
    def get(self, i: int) -> np.ndarray:
        """One epoch as a dense ``(n_channels, n_i)`` array."""

    @abc.abstractmethod
    def select_epochs(self, idx) -> RaggedStore: ...

    @abc.abstractmethod
    def select_channels(self, idx) -> RaggedStore: ...

    # -- shared helpers -------------------------------------------------
    def __iter__(self):
        for i in range(len(self)):
            yield self.get(i)

    def to_dense(self, pad_value: float = np.nan) -> np.ndarray:
        """Right-pad to ``(n_epochs, n_channels, max(lengths))``.

        This is the ndarray boundary. Ragged for storage, indexing and IO;
        dense at the point of computation. It is deliberately explicit -- no
        user-facing analysis method calls it implicitly.
        """
        lengths = self.lengths
        out = np.full(
            (len(self), self.n_channels, int(lengths.max())),
            pad_value,
            dtype=np.float64,
        )
        for i in range(len(self)):
            out[i, :, : lengths[i]] = self.get(i)
        return out

    def __repr__(self) -> str:
        n = len(self)
        if n:
            lo, hi = int(self.lengths.min()), int(self.lengths.max())
            span = f"{lo}-{hi}" if lo != hi else f"{lo}"
        else:
            span = "0"
        return (
            f"<{type(self).__name__} | {n} epochs, {self.n_channels} channels, "
            f"{span} samples, {self.nbytes / 1e6:.1f} MB>"
        )


def _validate(blocks: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Check the only-time-is-ragged invariant and normalise dtype."""
    if len(blocks) == 0:
        raise ValueError("Need at least one epoch.")
    out = []
    n_channels = None
    for i, b in enumerate(blocks):
        b = np.asarray(b, dtype=np.float64)
        if b.ndim != 2:
            raise ValueError(
                f"Epoch {i} has ndim={b.ndim}; expected 2-D (n_channels, n_times)."
            )
        if n_channels is None:
            n_channels = b.shape[0]
        elif b.shape[0] != n_channels:
            raise ValueError(
                f"Epoch {i} has {b.shape[0]} channels but epoch 0 has "
                f"{n_channels}. Only the time axis may be ragged."
            )
        if b.shape[1] == 0:
            raise ValueError(f"Epoch {i} has zero samples.")
        out.append(b)
    return out, int(n_channels)


def _as_index(idx, n: int) -> np.ndarray:
    """Normalise slice / bool mask / int / sequence to an integer index array."""
    if isinstance(idx, slice):
        return np.arange(n)[idx]
    arr = np.asarray(idx)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError(f"Boolean mask has shape {arr.shape}, expected ({n},).")
        return np.flatnonzero(arr)
    return np.atleast_1d(arr).astype(int)


class ListStore(RaggedStore):
    """``list[np.ndarray]`` plus an offsets array.

    Zero new dependencies, channel-major (matching MNE's ``(n_channels,
    n_times)`` convention), and dense conversion is a straight copy with no
    transpose. This is the baseline every other backend has to beat.
    """

    name = "list"

    def __init__(self, blocks: list[np.ndarray], n_channels: int):
        self._blocks = blocks
        self._n_channels = n_channels
        self._lengths = np.array([b.shape[1] for b in blocks], dtype=np.int64)

    @classmethod
    def from_list(cls, blocks):
        blocks, n_channels = _validate(blocks)
        return cls(blocks, n_channels)

    @property
    def lengths(self):
        return self._lengths

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def nbytes(self):
        return int(sum(b.nbytes for b in self._blocks))

    @property
    def offsets(self) -> np.ndarray:
        """Cumulative sample offsets, ``(n_epochs + 1,)``."""
        return np.concatenate([[0], np.cumsum(self._lengths)])

    def get(self, i):
        return self._blocks[i]

    def select_epochs(self, idx):
        sel = _as_index(idx, len(self))
        return ListStore([self._blocks[i] for i in sel], self._n_channels)

    def select_channels(self, idx):
        sel = _as_index(idx, self._n_channels)
        return ListStore([b[sel] for b in self._blocks], len(sel))


class PaddedStore(RaggedStore):
    """Dense ``(n_epochs, n_channels, n_max)`` plus explicit lengths.

    The route PR #12315 took. Kept as a benchmark baseline and as the
    interchange/IO format -- not as the working container. Note that validity
    lives in ``lengths``, never inside the array: a ``np.ma.MaskedArray`` is
    silently stripped by SciPy (verified on 1.17.1), so keeping the validity
    information *outside* the payload is the only safe option.
    """

    name = "padded"

    def __init__(self, data: np.ndarray, lengths: np.ndarray):
        self._data = data
        self._lengths = lengths.astype(np.int64)

    @classmethod
    def from_list(cls, blocks):
        blocks, n_channels = _validate(blocks)
        lengths = np.array([b.shape[1] for b in blocks], dtype=np.int64)
        data = np.full((len(blocks), n_channels, int(lengths.max())), np.nan)
        for i, b in enumerate(blocks):
            data[i, :, : b.shape[1]] = b
        return cls(data, lengths)

    @property
    def lengths(self):
        return self._lengths

    @property
    def n_channels(self):
        return self._data.shape[1]

    @property
    def nbytes(self):
        return int(self._data.nbytes)

    def get(self, i):
        return self._data[i, :, : self._lengths[i]]

    def select_epochs(self, idx):
        sel = _as_index(idx, len(self))
        lengths = self._lengths[sel]
        return PaddedStore(self._data[sel][:, :, : int(lengths.max())], lengths)

    def select_channels(self, idx):
        sel = _as_index(idx, self.n_channels)
        return PaddedStore(self._data[:, sel, :], self._lengths)

    def to_dense(self, pad_value=np.nan):
        out = self._data.copy()
        if not np.isnan(pad_value):
            for i, n in enumerate(self._lengths):
                out[i, :, n:] = pad_value
        return out


class AwkwardStore(RaggedStore):
    """``awkward.Array`` with layout ``n_epochs * var * n_channels * float64``.

    The layout matters and is easy to get wrong. ``n_epochs * n_channels *
    var`` -- the intuitive reading of "epochs of channels of ragged time" --
    does **not** structurally prevent channel 0 and channel 1 of the same epoch
    from having different lengths. Putting ``var`` at the epoch level and
    keeping channels as the regular inner dimension is what actually enforces
    the invariant.

    The cost is that this is time-major ``(epoch, time, channel)`` while MNE is
    channel-major, so every dense conversion pays a transpose. Whether that
    cost is worth the memory saving is exactly what the benchmark measures.
    """

    name = "awkward"

    def __init__(self, array, n_channels: int):
        import awkward as ak

        self._array = array
        self._n_channels = n_channels
        self._lengths = np.asarray(ak.num(array, axis=1), dtype=np.int64)

    @classmethod
    def from_list(cls, blocks):
        blocks, n_channels = _validate(blocks)
        try:
            import awkward as ak
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "The 'awkward' backend needs the awkward package: pip install awkward"
            ) from exc
        # Build via a flat (total_samples, n_channels) buffer and unflatten.
        #
        # The obvious construction -- ak.Array([b.T for b in blocks]) -- looks
        # right and is wrong: awkward infers `n_epochs * var * var * float64`,
        # so the *channel* dimension is ragged too and the invariant this class
        # exists to enforce is silently gone. It is also ~5000x slower to build
        # and ~1000x slower to index, because every epoch becomes a separate
        # Python-level list rather than a view on one contiguous buffer.
        #
        # Going through ak.from_numpy(..., regulararray=True) keeps channels as
        # a RegularArray, giving `n_epochs * var * n_channels * float64`.
        flat = np.concatenate([b.T for b in blocks], axis=0)  # (total, n_ch)
        counts = [b.shape[1] for b in blocks]
        array = ak.unflatten(
            ak.from_numpy(flat, regulararray=True), counts, axis=0
        )
        return cls(array, n_channels)

    @property
    def lengths(self):
        return self._lengths

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def nbytes(self):
        return int(self._array.nbytes)

    @property
    def typestr(self) -> str:
        """Awkward type string, quoted verbatim in the benchmark write-up."""
        return str(self._array.type)

    def get(self, i):
        return np.asarray(self._array[i]).T

    def select_epochs(self, idx):
        sel = _as_index(idx, len(self))
        return AwkwardStore(self._array[sel], self._n_channels)

    def select_channels(self, idx):
        sel = _as_index(idx, self._n_channels)
        return AwkwardStore(self._array[:, :, sel], len(sel))


BACKENDS: dict[str, type[RaggedStore]] = {
    "list": ListStore,
    "padded": PaddedStore,
    "awkward": AwkwardStore,
}


def get_backend(name: str) -> type[RaggedStore]:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {sorted(BACKENDS)}"
        ) from None
