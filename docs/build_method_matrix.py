"""Generate docs/05-method-matrix.md.

The classification is written here as data and joined against the *live*
``mne.BaseEpochs`` API, so the "all N methods covered" claim is checkable
rather than asserted: any method MNE gains or renames shows up as UNCLASSIFIED
the next time this runs.

Run:  python docs/build_method_matrix.py
"""

from __future__ import annotations

import pathlib

import mne

# ---------------------------------------------------------------- classes
CLASSES = {
    "bookkeeping": (
        "Bookkeeping",
        "No time axis involved. Works unchanged; raggedness is irrelevant.",
    ),
    "spatial": (
        "Spatial",
        "Acts on the channel axis only. Works unchanged.",
    ),
    "ragged": (
        "Naturally ragged",
        "Mathematically per-trial. Maps over epochs with no kernel rewrite; "
        "parity against per-epoch MNE is asserted in `tests/test_parity_with_mne.py`.",
    ),
    "policy": (
        "Ragged with declared policy",
        "Supported, but a cross-trial semantic must be stated explicitly "
        "(sample- vs epoch-weighting, common frequency grid).",
    ),
    "ragged_out": (
        "Ragged output",
        "Consumes ragged input and returns a ragged-time representation.",
    ),
    "needs_axis": (
        "Requires a common time axis",
        "Raises on ragged input and names the alignment options. **Not** a "
        "silent crop -- this is the answer to agramfort's time-dependent-`nave` "
        "objection on #12315.",
    ),
    "transform": (
        "Length-changing transformation",
        "Legitimately changes the number of samples. Stays ragged.",
    ),
    "io": (
        "IO",
        "Needs a container format decision; padding plus an explicit lengths "
        "array is the interchange form (never a masked array -- SciPy drops "
        "the mask).",
    ),
}

# ------------------------------------------------------------ the matrix
M: dict[str, tuple[str, str]] = {
    # -- bookkeeping ----------------------------------------------------
    "add_annotations_to_metadata": ("bookkeeping", "Already yields per-epoch landmark latencies relative to epoch t=0 -- the exact input `landmark_warp` wants. Reuse, do not rebuild."),
    "get_annotations_per_epoch": ("bookkeeping", "Same: per-epoch `(onset, duration, description)`, onset already epoch-relative."),
    "anonymize": ("bookkeeping", ""),
    "copy": ("bookkeeping", ""),
    "load_data": ("bookkeeping", ""),
    "next": ("bookkeeping", "Iteration yields per-epoch arrays of differing length."),
    "drop": ("bookkeeping", ""),
    "drop_bad": ("bookkeeping", "`reject`/`flat` are per-epoch amplitude criteria; duration-independent."),
    "drop_log_stats": ("bookkeeping", ""),
    "reset_drop_log_selection": ("bookkeeping", ""),
    "equalize_event_counts": ("bookkeeping", "Equalises counts, not durations."),
    "set_annotations": ("bookkeeping", ""),
    "set_meas_date": ("bookkeeping", ""),
    "get_channel_types": ("bookkeeping", ""),
    "get_montage": ("bookkeeping", ""),
    "set_montage": ("bookkeeping", ""),
    "set_channel_types": ("bookkeeping", ""),
    "plot_drop_log": ("bookkeeping", ""),
    "plot_sensors": ("bookkeeping", ""),
    "plot_projs_topomap": ("bookkeeping", ""),
    "add_proj": ("bookkeeping", ""),
    "del_proj": ("bookkeeping", ""),
    # -- spatial --------------------------------------------------------
    "pick": ("spatial", ""),
    "pick_channels": ("spatial", ""),
    "pick_types": ("spatial", ""),
    "drop_channels": ("spatial", ""),
    "rename_channels": ("spatial", ""),
    "reorder_channels": ("spatial", ""),
    "add_channels": ("spatial", "Requires matching per-epoch lengths between the two objects."),
    "add_reference_channels": ("spatial", ""),
    "set_eeg_reference": ("spatial", "Verified against MNE per-epoch."),
    "interpolate_bads": ("spatial", "Spherical-spline interpolation is a spatial operator applied per time point."),
    "interpolate_to": ("spatial", "Same."),
    "as_type": ("spatial", ""),
    # -- naturally ragged ------------------------------------------------
    "apply_baseline": ("ragged", "Per-epoch window. Verified against MNE per-epoch."),
    "apply_function": ("ragged", "The generic escape hatch; `map_epochs` is its ragged twin."),
    "apply_hilbert": ("ragged", "Zero-pad length is duration-dependent (MNE's own `next_fast_len`, 2/3/5-smooth). Verified against MNE per-epoch."),
    "apply_proj": ("ragged", ""),
    "filter": ("ragged", "Edge effects are duration-dependent -- true of fixed-length epoching too, just usually invisible. Filter `Raw` before epoching where possible. Verified against MNE per-epoch (note `Epochs.filter` uses `pad='edge'`, not `filter_data`'s `'reflect_limited'`)."),
    "savgol_filter": ("ragged", "Window must fit the shortest epoch."),
    "subtract_evoked": ("needs_axis", "Subtracts a common-axis Evoked from each epoch, so it needs one."),
    "get_data": ("ragged", "`representation={'ragged','dense','concatenated'}` is explicit. No default is right for every caller."),
    "to_data_frame": ("ragged", "Long format already expresses ragged data: one row per epoch x time x channel. Wide format needs a common axis."),
    "time_as_index": ("ragged", "Needs a per-epoch variant: `time_as_index(t, epoch=i)`."),
    "plot": ("ragged", "The browser shows epochs sequentially, so per-epoch axes are natural. See #10367 on epoch time-axis labelling."),
    # -- policy ---------------------------------------------------------
    "compute_psd": ("policy", "Two policies: the common frequency grid (fixing `n_fft` caps resolution at the shortest epoch) and sample- vs epoch-weighting. This is where PR #12315 started."),
    "plot_psd": ("policy", "Inherits `compute_psd`'s policies."),
    "plot_psd_topo": ("policy", "Inherits `compute_psd`'s policies."),
    "plot_psd_topomap": ("policy", "Inherits `compute_psd`'s policies."),
    # -- ragged output ---------------------------------------------------
    "compute_tfr": ("ragged_out", "`average=False` returns a `RaggedEpochsTFR`: frequency axis common, time axis ragged. `average=True` requires alignment first -- and that alignment must happen in the TF domain (see V1)."),
    # -- needs a common axis ----------------------------------------------
    "average": ("needs_axis", "Raises, naming crop / pad / normalise / landmark-warp. Under padding `nave` becomes a function of time; `pad()` returns that vector rather than hiding it behind a scalar."),
    "standard_error": ("needs_axis", "Same, and the time-dependent `nave` propagates directly into the error."),
    "iter_evoked": ("needs_axis", "Each yielded Evoked would carry a different time axis."),
    "plot_image": ("needs_axis", "The image axis is the shared time axis."),
    "plot_topo_image": ("needs_axis", "Same."),
    # -- length-changing --------------------------------------------------
    "crop": ("transform", "Per-epoch tmin/tmax; stays ragged. Absolute-time cropping can also *create* raggedness from uniform epochs."),
    "decimate": ("transform", "Integer factor per epoch; ragged in, ragged out."),
    "resample": ("transform", "Changes the physical sampling rate. Distinct from duration normalisation -- resampling preserves duration, warping does not. Conflating the two is what the fake-`sfreq` hack did."),
    "shift_time": ("transform", "Per-epoch shift is exactly issue #5794, which this design covers as the rectangular special case."),
    # -- IO ---------------------------------------------------------------
    "save": ("io", "FIF assumes a rectangular epoch block. Options: pad + store lengths, or a new tag. Padding is what mne-connectivity #142 shipped for its ragged case."),
    "export": ("io", "Same, plus each external format's own constraints."),
}

# ------------------------------------------------------ external, not on Epochs
EXTERNAL = [
    ("mne.decoding.*", "Requires rectangular tensor", "Sliding-window and time-generalisation estimators need `(n_epochs, n_channels, n_times)`. Align or extract features first."),
    ("mne.stats.permutation_cluster_*", "Requires rectangular tensor", "Clustering is over an adjacency structure on a common time axis."),
    ("mne.compute_covariance", "Ragged with declared policy", "Sample- vs epoch-weighting, and it propagates into the inverse operator's noise model."),
    ("mne.preprocessing.ICA.fit", "Ragged with declared policy", "Already concatenates to `(n_channels, n_epochs * n_times)`, so ragged ICA is natural. The open question is whether a 5 s trial should carry 5x the influence of a 1 s trial."),
    ("mne.minimum_norm.apply_inverse_epochs", "Ragged output", "Per-epoch source estimates of differing length."),
]


def main():
    methods = sorted(
        n for n in dir(mne.BaseEpochs)
        if not n.startswith("_") and callable(getattr(mne.BaseEpochs, n, None))
    )
    props = sorted(
        n for n in dir(mne.BaseEpochs)
        if not n.startswith("_") and not callable(getattr(mne.BaseEpochs, n, None))
    )
    missing = [m for m in methods if m not in M]
    extra = [m for m in M if m not in methods]

    L = []
    A = L.append
    A("# Method matrix -- every public `BaseEpochs` API under raggedness\n")
    A("*Generated by `docs/build_method_matrix.py` against the installed MNE.*")
    A(f"*MNE {mne.__version__}: {len(methods)} public methods, "
      f"{len(props)} public properties.*\n")
    A("The historical objection to variable-length epochs is kingjr's on #3533: "
      "supporting them means special-casing everything downstream -- TFR, PSD, "
      "plotting, `times`, covariance, ICA, `nave`, statistics. That objection is "
      "correct if you try to make every method ragged-aware one at a time.\n")
    A("The answer is to classify by **mathematical meaning** instead. Most of the "
      "API turns out not to care about duration at all, a well-defined minority "
      "needs a stated policy, and a small set genuinely requires a common time "
      "axis and should say so rather than guess.\n")

    A("## Summary\n")
    A("| Class | Methods | Meaning |")
    A("|---|---:|---|")
    for key, (label, desc) in CLASSES.items():
        n = sum(1 for k, (c, _) in M.items() if c == key)
        A(f"| **{label}** | {n} | {desc} |")
    A("")
    n_free = sum(1 for _, (c, _) in M.items() if c in ("bookkeeping", "spatial", "ragged", "transform"))
    A(f"**{n_free} of {len(methods)} methods ({100 * n_free // len(methods)}%) need no "
      "cross-trial decision at all.** Only "
      f"{sum(1 for _, (c, _) in M.items() if c == 'needs_axis')} genuinely require a "
      "common time axis, and "
      f"{sum(1 for _, (c, _) in M.items() if c == 'policy')} need a stated policy. "
      "That is the scope of the problem, and it is far smaller than the "
      "Pandora's-box framing suggests.\n")

    for key, (label, desc) in CLASSES.items():
        rows = sorted((k, v[1]) for k, v in M.items() if v[0] == key)
        A(f"## {label}\n")
        A(f"{desc}\n")
        A("| Method | Notes |")
        A("|---|---|")
        for name, note in rows:
            A(f"| `{name}` | {note} |")
        A("")

    A("## Properties\n")
    A("| Property | Behaviour |")
    A("|---|---|")
    notes = {
        "times": "**Raises `RaggedTimesError`** unless every epoch shares an axis. Never a silent fallback to the shortest common interval -- that is the easiest way for this feature to produce quiet scientific errors. The message names `durations`, `get_times()` and `align_time()`.",
        "tmin": "Becomes `(n_epochs,)`.",
        "tmax": "Becomes `(n_epochs,)`.",
        "annotations": "Unchanged.",
        "ch_names": "Unchanged.",
        "compensation_grade": "Unchanged.",
        "filename": "Unchanged.",
        "metadata": "Unchanged, and the natural home for per-trial landmark latencies.",
        "proj": "Unchanged.",
    }
    for p in props:
        A(f"| `{p}` | {notes.get(p, 'Unchanged.')} |")
    A("")
    A("New: `durations` `(n_epochs,)`, `lengths` `(n_epochs,)`, `is_uniform` `bool`, "
      "`alignment` (`AlignmentRecord | None`).\n")

    A("## Beyond `Epochs`\n")
    A("| API | Class | Notes |")
    A("|---|---|---|")
    for name, cls, note in EXTERNAL:
        A(f"| `{name}` | {cls} | {note} |")
    A("")

    if missing:
        A("## UNCLASSIFIED -- regenerate this file\n")
        for m in missing:
            A(f"- `{m}`")
        A("")
    if extra:
        A("## Classified but absent from this MNE version\n")
        for m in extra:
            A(f"- `{m}`")
        A("")

    out = pathlib.Path(__file__).parent / "05-method-matrix.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  {len(methods)} methods, {len(M)} classified, "
          f"{len(missing)} unclassified, {len(extra)} stale")
    return missing, extra


if __name__ == "__main__":
    main()
