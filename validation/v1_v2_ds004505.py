"""V1 and V2: validate variable-duration Epochs against the ds004505 run.

V1  Round-trip fidelity. Every epoch the container returns must be bit-identical
    to the raw slice it came from. Exact equality, not a tolerance.

V2  Numerical equivalence with array job 55417405 (2026-08-18). The real swing
    cycles routed through the container, through the same Morlet transform and
    warp, must reproduce the saved sub-NN_ersp.npz.

Two notes on how this is done.

*Why the transform runs once, not twice.* V1 establishes that the container's
array and the production slice are byte-identical. Transforming both would then
compare NumPy against itself rather than testing anything about the container,
at twice the cost. So V1 tests the data and V2 tests the chain.

*Why it accumulates.* Holding every warped cycle needs
``n_cycles * n_channels * n_freqs * n_grid * 8`` bytes, which is ~17 GB for one
condition of one subject. Voltage rejection only needs the peak of each raw
segment, which is available without a transform, so pass 1 collects peaks and
pass 2 transforms just the retained cycles into a running sum. Same arithmetic,
bounded memory.

Production indexes with ``int((t0 - pad) * sf)``, which truncates, while Epochs
rounds. Events are derived from production's own ``s0`` so that both read exactly
the same samples and the comparison is about the container.

Run:
    python validation/v1_v2_ds004505.py --stage-dir D:/ds004505-local \\
        --ref-dir D:/ds004505-local/ref --subjects sub-02 sub-03 sub-05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")

# -- production constants, copied verbatim from eneuro_merged_ersp.py ------
FREQS = np.linspace(3.0, 40.0, 60)
N_CYCLES = np.linspace(3.0, 64.0, FREQS.size)
N_GRID = 201
CYCLE_MIN, CYCLE_MAX = 0.6, 4.0
EPOCH_REJECT_PCT = 10.0
PAD = 1.0


def build_cycles(appear, hit):
    """Three-point cycles: appearance, contact, next appearance.

    Parameters
    ----------
    appear : array
        Times of the appearance events in seconds.
    hit : array
        Times of the participant's contacts in seconds.

    Returns
    -------
    cycles : array, shape (n, 3)
        Rows of ``[t_appear, t_hit, t_next]``.
    """
    rows = []
    for a, b in zip(appear[:-1], appear[1:]):
        dur = b - a
        if not (CYCLE_MIN <= dur <= CYCLE_MAX):
            continue
        inside = hit[(hit > a) & (hit < b)]
        if inside.size != 1:
            continue
        lat = inside[0] - a
        if not (0.05 < lat < dur - 0.05):
            continue
        rows.append((a, inside[0], b))
    return np.asarray(rows) if rows else np.empty((0, 3))


def reject_high_voltage(peaks, pct):
    """Return indices of the epochs to keep after voltage rejection.

    Parameters
    ----------
    peaks : array
        Per-epoch peak absolute voltage.
    pct : float
        Percentage of epochs to drop.

    Returns
    -------
    keep : array of int
        Sorted indices of retained epochs.
    """
    peaks = np.asarray(peaks, dtype=float)
    if peaks.size == 0:
        return np.array([], dtype=int)
    n_keep = max(1, int(round(peaks.size * (1.0 - pct / 100.0))))
    return np.sort(np.argsort(peaks)[:n_keep])


def production_windows(cycles, sfreq, n_total):
    """Return the sample windows the production script would read.

    Parameters
    ----------
    cycles : array, shape (n, 3)
        Cycle bounds in seconds.
    sfreq : float
        Sampling frequency in Hz.
    n_total : int
        Number of samples in the recording.

    Returns
    -------
    windows : list of tuple
        ``(s0, s1, t0, t_hit, t1)`` for each retained cycle.
    """
    out = []
    for t0, t_hit, t1 in cycles:
        s0, s1 = int((t0 - PAD) * sfreq), int((t1 + PAD) * sfreq)
        if s0 < 0 or s1 >= n_total:
            continue
        out.append((s0, s1, t0, t_hit, t1))
    return out


def _inside_mask(n_samples, sfreq, dur):
    """Return the mask selecting the cycle proper out of a padded window."""
    times = np.arange(n_samples) / sfreq - PAD
    return times, (times >= 0) & (times <= dur)


def build_variable_epochs(raw, windows):
    """Build variable-duration Epochs covering production's exact windows.

    Parameters
    ----------
    raw : instance of Raw
        The cleaned continuous data.
    windows : list of tuple
        Output of :func:`production_windows`.

    Returns
    -------
    epochs : instance of mne.Epochs
        One epoch per window.
    """
    sfreq = float(raw.info["sfreq"])
    n_pad = int(round(PAD * sfreq))
    events, tmin, tmax = [], [], []
    for s0, s1, *_ in windows:
        # place the event so that Epochs reads exactly [s0, s1)
        events.append([s0 + n_pad + raw.first_samp, 0, 1])
        n = s1 - s0
        tmin.append(-n_pad / sfreq)
        tmax.append((-n_pad + n - 1) / sfreq)
    return mne.Epochs(
        raw,
        np.asarray(events, dtype=int),
        tmin=np.asarray(tmin),
        tmax=np.asarray(tmax),
        baseline=None,
        preload=True,
        proj=False,
        reject_by_annotation=False,
        verbose=False,
    )


def ersp_from_blocks(blocks, windows, sfreq, hit_frac):
    """Compute the mean warped log-power, accumulating rather than storing.

    Parameters
    ----------
    blocks : list of array
        Per-cycle padded data, ``(n_channels, n_times_i)``.
    windows : list of tuple
        Matching entries from :func:`production_windows`.
    sfreq : float
        Sampling frequency in Hz.
    hit_frac : float
        Group warp anchor, as a fraction of the cycle.

    Returns
    -------
    mean_db : array, shape (n_channels, n_freqs, N_GRID)
        Mean of ``10 * log10(power)`` over retained cycles.
    n_kept : int
        Number of cycles contributing.
    """
    grid = np.linspace(0.0, 1.0, N_GRID)

    # pass 1: peaks and usability, both available without a transform
    peaks, usable = [], []
    for block, (_, _, t0, t_hit, t1) in zip(blocks, windows):
        times, inside = _inside_mask(block.shape[1], sfreq, t1 - t0)
        if inside.sum() < 10:
            continue
        usable.append((block, times, inside, t0, t_hit, t1))
        peaks.append(float(np.abs(block[:, inside]).max()))
    if not usable:
        return None, 0

    keep = (
        reject_high_voltage(np.asarray(peaks), EPOCH_REJECT_PCT)
        if len(usable) > 1
        else np.arange(len(usable))
    )

    # pass 2: transform only what is retained, summing in place
    total = None
    for idx in keep:
        block, times, inside, t0, t_hit, t1 = usable[idx]
        power = mne.time_frequency.tfr_array_morlet(
            block[np.newaxis],
            sfreq=sfreq,
            freqs=FREQS,
            n_cycles=N_CYCLES,
            output="power",
            zero_mean=True,
            verbose=False,
        )[0]
        src = np.interp(grid, [0.0, hit_frac, 1.0], [0.0, t_hit - t0, t1 - t0])
        warped = np.stack(
            [
                [
                    np.interp(src, times[inside], power[c, f, inside])
                    for f in range(power.shape[1])
                ]
                for c in range(power.shape[0])
            ]
        )
        np.log10(warped, out=warped)
        warped *= 10.0
        total = warped if total is None else total + warped
    return total / len(keep), int(len(keep))


def run_subject(subject, stage_dir, ref_dir, report):
    """Validate one subject and append its result to ``report``."""
    stage_dir, ref_dir = Path(stage_dir), Path(ref_dir)
    raw_path = stage_dir / f"{subject}_clean_raw.fif"
    if not raw_path.exists():
        print(f"[{subject}] SKIP: {raw_path} not present", flush=True)
        return

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    ev = np.load(stage_dir / f"{subject}_events.npz")
    meta = json.loads((ref_dir / f"{subject}_ersp.json").read_text())
    hit_frac = float(meta["hit_frac_applied"])
    ref = np.load(ref_dir / f"{subject}_ersp.npz")
    sfreq = float(raw.info["sfreq"])
    raw_data = raw.get_data()

    print(f"\n=== {subject} ===", flush=True)
    print(
        f"  raw {len(raw.ch_names)} ch @ {sfreq:g} Hz, {raw.times[-1]:.0f} s "
        f"| hit_frac={hit_frac}",
        flush=True,
    )

    entry = {"subject": subject, "hit_frac": hit_frac, "sfreq": sfreq}
    maps, ns = {}, {}
    v1_ok, v1_checked = True, 0

    for cond, appear in (("machine", "machine_feed"), ("human", "researcher_hit")):
        cycles = build_cycles(ev[appear], ev["participant_hit"])
        if cycles.size == 0:
            continue
        windows = production_windows(cycles, sfreq, raw_data.shape[1])
        epochs = build_variable_epochs(raw, windows)
        blocks = epochs.get_data()
        assert len(blocks) == len(windows), (len(blocks), len(windows))

        # V1: the container returns exactly the samples production would read
        for block, (s0, s1, *_) in zip(blocks, windows):
            if not np.array_equal(block, raw_data[:, s0:s1]):
                v1_ok = False
                break
            v1_checked += 1

        mean_db, n_kept = ersp_from_blocks(blocks, windows, sfreq, hit_frac)
        if mean_db is not None:
            maps[cond], ns[cond] = mean_db, n_kept
        entry[f"n_{cond}_cycles"] = int(len(cycles))
        del epochs, blocks

    entry["v1_exact"] = bool(v1_ok)
    entry["v1_cycles_checked"] = v1_checked
    print(
        f"  V1 round-trip : {'EXACT' if v1_ok else 'MISMATCH'} "
        f"over {v1_checked} cycles",
        flush=True,
    )

    # pooled whole-cycle baseline across conditions, as production does
    if maps:
        base = sum(maps[k].mean(axis=2) * ns[k] for k in maps) / sum(ns.values())
        for k in maps:
            maps[k] = maps[k] - base[:, :, None]

    for cond in sorted(maps):
        key = f"map_{cond}"
        if key not in ref:
            continue
        reference = ref[key]
        delta = float(np.abs(maps[cond] - reference).max())
        scale = float(np.abs(reference).max())
        entry[f"{cond}_n_ref"] = int(ref[f"n_{cond}"])
        entry[f"{cond}_n_local"] = ns[cond]
        entry[f"{cond}_max_abs_vs_ref_db"] = delta
        entry[f"{cond}_ref_scale_db"] = scale
        print(
            f"  V2 {cond:8s}: n ref={int(ref[f'n_{cond}'])} local={ns[cond]} "
            f"| max |ours - cluster| {delta:.3e} dB (scale {scale:.1f} dB)",
            flush=True,
        )

    report.append(entry)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", default="validation/v1_v2_report.json")
    ap.add_argument(
        "--tol-db",
        type=float,
        default=1e-9,
        help="max acceptable |ours - cluster| in dB; above this is a failure",
    )
    args = ap.parse_args(argv)

    print(f"mne {mne.__version__} from {Path(mne.__file__).parent}", flush=True)
    report = []
    for subject in args.subjects:
        run_subject(subject, args.stage_dir, args.ref_dir, report)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}", flush=True)

    if not report:
        print("NOTHING VALIDATED")
        return 1
    v1 = all(e["v1_exact"] for e in report)
    worst = max(
        (v for e in report for k, v in e.items() if k.endswith("_max_abs_vs_ref_db")),
        default=float("inf"),
    )
    counts_ok = all(
        e.get(f"{c}_n_ref") == e.get(f"{c}_n_local")
        for e in report
        for c in ("machine", "human")
        if f"{c}_n_ref" in e
    )
    print(f"\nV1 exact on all subjects   : {v1}")
    print(f"epoch counts match reference: {counts_ok}")
    print(f"worst |ours - cluster|      : {worst:.3e} dB (tolerance {args.tol_db:g})")
    return 0 if (v1 and counts_ok and worst <= args.tol_db) else 1


if __name__ == "__main__":
    sys.exit(main())
