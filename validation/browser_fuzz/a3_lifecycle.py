"""A3 — selection / drop lifecycle probes for variable-duration epochs.

Every epoch is tagged: channel 0 of source epoch ``i`` is filled with ``i``, so
after any reorder / subset / drop we can recover *which* source epoch each slot
holds and check that ``_tmin_per_epoch`` / ``_tmax_per_epoch`` moved with it.

Nothing here reads the object's own bookkeeping to build its expectation.
"""

import numpy as np

import mne
from mne import EpochsArray, create_info

SFREQ = 100.0
TAG_UNIT = 1e-6  # the flat tag amplitude, chosen to stay on an EEG scale


def make(lengths, tmins=None, n_channels=3, sfreq=SFREQ, event_ids=None, metadata=None):
    """Build ragged epochs whose channel 0 is a constant tag = source index."""
    n = len(lengths)
    if tmins is None:
        tmins = np.zeros(n)
    tmins = np.asarray(tmins, float)
    names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = create_info(names, sfreq, ["eeg"] * n_channels)
    rng = np.random.default_rng(0)
    arrays = []
    for i, L in enumerate(lengths):
        a = rng.standard_normal((n_channels, L)) * 1e-6
        a[0, :] = i * TAG_UNIT  # the tag: a flat channel naming its source epoch
        arrays.append(a)
    stride = int(max(lengths)) + int(sfreq) + 100
    if event_ids is None:
        event_ids = np.ones(n, int)
    event_ids = np.asarray(event_ids, int)
    events = np.column_stack(
        [np.arange(n) * stride + stride, np.zeros(n, int), event_ids]
    )
    eid = {f"e{v}": int(v) for v in np.unique(event_ids)}
    ep = EpochsArray(
        arrays,
        info,
        events=events,
        tmin=tmins,
        event_id=eid,
        baseline=None,
        metadata=metadata,
        verbose=False,
    )
    return ep


def tags(ep):
    """Which source epoch sits in each slot, read from the data itself."""
    data = ep._data
    return [int(round(float(d[0, 0]) / TAG_UNIT)) for d in data]


def lengths_of(ep):
    return [int(d.shape[-1]) for d in ep._data]


def audit(ep, label, expect_tags=None, expect_tmins=None, sfreq=SFREQ):
    """Return a list of violation strings for one epochs object."""
    out = []
    n_data = len(ep._data)
    for name, arr in (
        ("events", ep.events),
        ("selection", ep.selection),
        ("_tmin_per_epoch", ep._tmin_per_epoch),
        ("_tmax_per_epoch", ep._tmax_per_epoch),
    ):
        if arr is None:
            out.append(f"{label}: {name} is None while _data has {n_data}")
            continue
        if len(arr) != n_data:
            out.append(f"{label}: len({name})={len(arr)} but len(_data)={n_data}")
    if ep.metadata is not None and len(ep.metadata) != n_data:
        out.append(f"{label}: len(metadata)={len(ep.metadata)} but _data={n_data}")
    if out:
        return out

    # each epoch's declared span must match its own sample count
    for i in range(n_data):
        want = (ep._data[i].shape[-1] - 1) / sfreq
        got = ep._tmax_per_epoch[i] - ep._tmin_per_epoch[i]
        if not np.isclose(want, got, atol=1e-9):
            out.append(
                f"{label}: epoch {i} holds {ep._data[i].shape[-1]} samples "
                f"({want:.6f} s) but bounds say {got:.6f} s "
                f"[{ep._tmin_per_epoch[i]:.6f}, {ep._tmax_per_epoch[i]:.6f}]"
            )
    got_tags = tags(ep)
    if expect_tags is not None and got_tags != list(expect_tags):
        out.append(f"{label}: slots hold source epochs {got_tags}, want {expect_tags}")
    if expect_tmins is not None:
        if not np.allclose(ep._tmin_per_epoch, expect_tmins, atol=1e-9):
            out.append(
                f"{label}: _tmin_per_epoch {np.asarray(ep._tmin_per_epoch)} "
                f"want {np.asarray(expect_tmins)}"
            )
    return out


def banner(s):
    print(f"\n===== {s} =====")
