"""V4 -- reproduce the poster ERSP through the new API.

**Read this before assuming the production pipeline is wrong.** The current
cluster code, `/scratch/sesma/scripts/eneuro_merged_ersp.py` (array job 55417405,
2026-08-18), already gets both methodological points right:

    p   = tfr_array_morlet(padded_slice, ...)                    # TFR first
    src = np.interp(grid, [0, hit_frac, 1], [0, t_hit - t0, dur]) # then warp

That is TFR-domain warping to a *group* median anchor -- the same thing EEGLAB's
`newtimef` does. The C1/C2 corrections in this repo were written against the
mne-mobi `old` branch (`channel_tfr_general.warp_sig`), which warps the signal
first and spaces anchors uniformly. That code is two generations stale.

What this script is for, then, is not "fix the production pipeline". It is:

  1. Show the same analysis expressed through the proposed API, so the design is
     judged against real work rather than a toy.
  2. Quantify what the stale mne-mobi path would have cost, since that is the
     failure mode a naive `time_warp_epochs` in MNE core would ship by default.

Anchor values from the completed run (not from the paper):

    hit_frac_applied  0.385     <- THIS COHORT's pooled median, all 24 subjects
    paper_hit_frac    0.3326    <- Studnicki & Ferris (640 ms of 1924 ms)
    cohort median cycle  ~1.92 s   vs paper 1.924 s
    per-subject hit_frac range  32.6-47.9%

The poster lists 38.5% as a *deviation from* the paper, not as the paper's
value. `--contact-pct` therefore defaults to the cohort value and the paper
value is reported alongside it.

Two-pass group anchoring, as the production code does it: run once without
`--hit-frac` to record each subject's own median and refuse to warp, pool those
across subjects, then run again with the group value. A single-subject median
used as the group anchor is a silent leak -- the production code carries a
comment about that having happened in an earlier runner.

The recordings are not in this repository. Run where they live:

    python validation/reproduce_poster_ersp.py \
        --raw /path/sub-01_task-walk_clean.fif \
        --out ./out --anchors RHS LTO LHS RTO
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mne
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragged_epochs import RaggedEpochs  # noqa: E402
from ragged_epochs._align import landmark_warp, resolve_target_landmarks  # noqa: E402
from ragged_epochs._tfr import compute_tfr, warp_tfr  # noqa: E402

mne.set_log_level("ERROR")


def find_cycles(raw, cycle_event="RHS", anchors=("LTO", "LHS", "RTO"),
                min_dur=0.8, max_dur=2.0):
    """RHS->RHS cycles with their internal anchors, in seconds.

    Same logic as mne-mobi's `find_complete_gait_cycles`, but cycles with a
    missing anchor are *reported* rather than silently warped with fewer
    anchors. mne-mobi's `keep_only_five_anchor` flag existed because mapping 4
    anchors and 5 anchors onto one target aligns different biomechanical events.
    """
    annot = raw.annotations
    t = {d: annot.onset[annot.description == d] - raw.first_time
         for d in set(annot.description)}
    if cycle_event not in t or len(t[cycle_event]) < 2:
        raise RuntimeError(f"Need >= 2 {cycle_event} events.")

    starts = t[cycle_event]
    onsets, durations, landmarks, dropped = [], [], [], {"duration": 0, "anchors": 0}

    for a, b in zip(starts[:-1], starts[1:]):
        dur = b - a
        if not (min_dur < dur < max_dur):
            dropped["duration"] += 1
            continue
        lm, ok = [0.0], True
        for name in anchors:
            hits = t.get(name, np.array([]))
            hits = hits[(hits > a) & (hits < b)]
            if hits.size == 0:
                ok = False
                break
            lm.append(float(hits[0] - a))
        if not ok:
            dropped["anchors"] += 1
            continue
        lm.append(float(dur))
        if np.any(np.diff(lm) <= 0):
            dropped["anchors"] += 1
            continue
        onsets.append(float(a))
        durations.append(float(dur))
        landmarks.append(np.array(lm))

    return np.array(onsets), np.array(durations), landmarks, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="cleaned continuous recording")
    ap.add_argument("--out", default="./v4_out")
    ap.add_argument("--cycle-event", default="RHS")
    ap.add_argument("--anchors", nargs="+", default=["LTO", "LHS", "RTO"])
    ap.add_argument("--picks", nargs="+", default=["Cz"])
    ap.add_argument("--fmin", type=float, default=3.0)
    ap.add_argument("--fmax", type=float, default=50.0)
    ap.add_argument("--n-freqs", type=int, default=40)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--contact-pct", type=float, default=38.5,
                    help="cohort pooled median contact position, %% of cycle "
                         "(the completed run used 38.5; the paper's value is 33.3)")
    ap.add_argument("--paper-contact-pct", type=float, default=33.264,
                    help="Studnicki & Ferris value, for reference in the report")
    ap.add_argument("--context", type=float, default=1.0,
                    help="seconds of surrounding data carried per cycle so no "
                         "wavelet taper straddles a landmark (production uses 1.0)")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_fif(args.raw, preload=True)
    raw.pick(args.picks)
    print(f"raw: {len(raw.ch_names)} channels @ {raw.info['sfreq']:g} Hz, "
          f"{raw.times[-1]:.0f} s")

    onsets, durations, landmarks, dropped = find_cycles(
        raw, args.cycle_event, tuple(args.anchors)
    )
    print(f"cycles: {len(onsets)} kept; dropped {dropped['duration']} on duration, "
          f"{dropped['anchors']} on missing/invalid anchors")
    print(f"durations {durations.min():.3f}-{durations.max():.3f} s "
          f"(median {np.median(durations):.3f})")

    ep = RaggedEpochs.from_raw(raw, onsets, durations, context=args.context)
    print(ep)

    names = (args.cycle_event, *args.anchors, f"{args.cycle_event}_next")
    dst = resolve_target_landmarks(landmarks, "median")
    pct = 100 * (dst - dst[0]) / (dst[-1] - dst[0])
    print("\nmedian anchor positions (% of cycle):")
    for n, p in zip(names, pct):
        print(f"  {n:<10s} {p:6.1f}%")
    uni = 100 * np.linspace(0, 1, len(dst))
    print("uniform (the C2 bug) would have put them at: "
          + ", ".join(f"{p:.1f}%" for p in uni))

    freqs = np.logspace(np.log10(args.fmin), np.log10(args.fmax), args.n_freqs)

    # ---- corrected path: TFR first, then warp the TF axis to the median ----
    tfr = compute_tfr(ep, freqs)
    tfr = tfr.apply_baseline(mode="logratio")     # per-cycle, Grandchamp-style
    new = warp_tfr(tfr, landmarks, target="median", n_points=args.n_points,
                   landmark_names=names)
    ersp_new = new.average()

    # ---- stale mne-mobi path, for contrast only -----------------------------
    # NOT what the cluster runs. This is `channel_tfr_general.warp_sig`: warp the
    # signal first, anchors spaced uniformly. Computed here to quantify what a
    # naive time_warp_epochs in MNE core would cost by default.
    old_ep = landmark_warp(ep, landmarks, target="uniform",
                           n_points=args.n_points, landmark_names=names)
    old_tfr = compute_tfr(old_ep, freqs).apply_baseline(mode="logratio")
    ersp_old = old_tfr.average()

    # ---- compare ----------------------------------------------------------
    def peak_freq(e):
        core = e[..., e.shape[-1] // 4 : 3 * e.shape[-1] // 4].mean(axis=-1)
        return float(freqs[np.argmax(core.mean(axis=0))])

    diff = float(np.abs(ersp_new - ersp_old).max())
    report = {
        "n_cycles": int(len(ep)),
        "dropped": dropped,
        "duration_s": {"min": float(durations.min()),
                       "max": float(durations.max()),
                       "median": float(np.median(durations))},
        "anchor_names": list(names),
        "anchor_pct_median": [float(p) for p in pct],
        "anchor_pct_uniform": [float(p) for p in uni],
        "cohort_contact_pct": args.contact_pct,
        "paper_contact_pct": args.paper_contact_pct,
        "tfr_context_s": args.context,
        "peak_freq_corrected_hz": peak_freq(ersp_new),
        "peak_freq_legacy_hz": peak_freq(ersp_old),
        "max_abs_ersp_difference_db": diff,
        "provenance": new.alignment.summary(),
    }
    (out / "v4_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out / "v4_ersp.npz",
        ersp_corrected=ersp_new, ersp_legacy=ersp_old,
        freqs=freqs, times=new.times,
        target_landmarks=dst, durations=durations,
    )

    print(f"\npeak frequency  corrected {report['peak_freq_corrected_hz']:.1f} Hz"
          f"   legacy {report['peak_freq_legacy_hz']:.1f} Hz")
    print(f"max |corrected - legacy| = {diff:.2f} dB")
    print(f"provenance: {report['provenance']}")
    print(f"\nwrote {out}/v4_report.json and v4_ersp.npz")
    print("If corrected and legacy differ materially, report the difference -- "
          "that is the finding, not a failure.")
    return report


if __name__ == "__main__":
    main()
