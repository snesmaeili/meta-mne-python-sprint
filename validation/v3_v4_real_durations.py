"""V3 and V4: as_fixed() and container cost on real ds004505 cycle durations.

V3  How fast does the contributing-epoch count decay across the union window,
    on real swing cycles rather than a synthetic spread? This is @agramfort's
    objection on #12315 measured instead of argued.

V4  Storage cost of keeping trials at their true durations versus padding them
    to the longest, using the measured duration distribution rather than the
    synthetic one in docs/06-container-benchmark.md.

Needs only the events files, not the raw recordings.

Run:
    python validation/v3_v4_real_durations.py --stage-dir D:/ds004505-local \\
        --subjects sub-02 sub-03 sub-05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from v1_v2_ds004505 import PAD, build_cycles  # noqa: E402

#: read from the cleaned recordings rather than assumed; the released tree
#: is 250 Hz while sourcedata/Merged is 500 Hz, and these are the cleaned
#: 250 Hz derivatives the ERSP job actually consumed
SFREQ = 250.0
N_CHANNELS = 120
BYTES = 8


def collect(stage_dir, subjects):
    """Return per-condition cycle durations for the given subjects.

    Parameters
    ----------
    stage_dir : path-like
        Directory holding ``{subject}_events.npz``.
    subjects : list of str
        Subject identifiers.

    Returns
    -------
    durations : dict
        Mapping of ``(subject, condition)`` to an array of durations.
    """
    out = {}
    for subject in subjects:
        path = Path(stage_dir) / f"{subject}_events.npz"
        if not path.exists():
            print(f"[{subject}] SKIP: {path} not present")
            continue
        ev = np.load(path)
        for cond, appear in (
            ("machine", "machine_feed"),
            ("human", "researcher_hit"),
        ):
            cycles = build_cycles(ev[appear], ev["participant_hit"])
            if cycles.size:
                out[(subject, cond)] = cycles[:, 2] - cycles[:, 0]
    return out


def support_curve(durations, sfreq=SFREQ, pad=PAD):
    """Return the number of epochs covering each sample of the union window.

    Parameters
    ----------
    durations : array
        Per-epoch duration in seconds.
    sfreq : float
        Sampling frequency in Hz.
    pad : float
        Context added on each side, as the production pipeline does.

    Returns
    -------
    n_contributing : array
        Count of epochs with real data at each time point.
    """
    lengths = np.round((durations + 2 * pad) * sfreq).astype(int) + 1
    n_max = int(lengths.max())
    return (np.arange(n_max)[None, :] < lengths[:, None]).sum(axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument(
        "--sfreq",
        type=float,
        default=SFREQ,
        help="sampling rate of the cleaned recordings, in Hz",
    )
    ap.add_argument("--out", default="validation/v3_v4_report.json")
    args = ap.parse_args(argv)

    sfreq = args.sfreq

    per_cond = collect(args.stage_dir, args.subjects)
    if not per_cond:
        print("NOTHING TO REPORT")
        return 1

    print("V3/V4 on real ds004505 swing cycles")
    print("=" * 78)
    rows = []
    for (subject, cond), durations in sorted(per_cond.items()):
        lengths = np.round(durations * sfreq).astype(int) + 1
        true_samples = int(lengths.sum())
        padded_samples = int(len(lengths) * lengths.max())
        waste = 100 * (1 - true_samples / padded_samples)
        support = support_curve(durations, sfreq=sfreq)
        half = int(np.argmax(support <= support[0] / 2))
        rows.append(
            dict(
                subject=subject,
                condition=cond,
                n_cycles=int(len(durations)),
                median_s=float(np.median(durations)),
                min_s=float(durations.min()),
                max_s=float(durations.max()),
                iqr_s=[
                    float(np.percentile(durations, 25)),
                    float(np.percentile(durations, 75)),
                ],
                true_samples=true_samples,
                padded_samples=padded_samples,
                padding_waste_pct=float(waste),
                support_start=int(support[0]),
                support_end=int(support[-1]),
                half_support_at_s=float(half / sfreq - PAD),
            )
        )
        print(
            f"{subject} {cond:8s} n={len(durations):5d} "
            f"median {np.median(durations):.3f} s "
            f"({durations.min():.2f}-{durations.max():.2f}) | "
            f"padding waste {waste:5.1f}% | "
            f"support {support[0]}->{support[-1]}, halves at "
            f"{half / sfreq - PAD:.2f} s"
        )

    pooled = np.concatenate(list(per_cond.values()))
    lengths = np.round(pooled * sfreq).astype(int) + 1
    true_samples = int(lengths.sum())
    padded_samples = int(len(lengths) * lengths.max())
    waste = 100 * (1 - true_samples / padded_samples)
    mb = N_CHANNELS * BYTES / 1e6

    print("-" * 78)
    print(
        f"pooled   n={len(pooled)} median {np.median(pooled):.3f} s "
        f"(paper 1.924) range {pooled.min():.2f}-{pooled.max():.2f} "
        f"IQR {np.percentile(pooled, 25):.2f}-{np.percentile(pooled, 75):.2f}"
    )
    print(
        f"storage  true {true_samples * mb:,.0f} MB vs padded "
        f"{padded_samples * mb:,.0f} MB at {N_CHANNELS} channels "
        f"-> {waste:.1f}% of the padded array is fill"
    )
    print(
        f"compare  the synthetic benchmark in docs/06-container-benchmark.md "
        f"assumed 32.5%"
    )

    summary = dict(
        per_condition=rows,
        pooled=dict(
            n_cycles=int(len(pooled)),
            median_s=float(np.median(pooled)),
            min_s=float(pooled.min()),
            max_s=float(pooled.max()),
            padding_waste_pct=float(waste),
            true_mb=float(true_samples * mb),
            padded_mb=float(padded_samples * mb),
            paper_cycle_s=1.924,
            sfreq=float(sfreq),
        ),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
