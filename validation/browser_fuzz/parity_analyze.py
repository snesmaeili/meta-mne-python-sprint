"""Post-hoc analysis of two parity recordings.

Answers the questions the summary diff raises: which specs a field diverges on,
how big the divergence is in real units, and whether any *downstream* field
(sample bounds, loaded data, plotted traces) moved with it.

    python parity_analyze.py <base_dir> <branch_dir> --backend matplotlib
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

# fields whose divergence would mean the reader sees different numbers
CRITICAL = (
    "epoch_color_ref",
    "ovb_bad_rects",
    "start",
    "stop",
    "loaded_data",
    "loaded_times",
    "data",
    "times",
    "boundary_times",
    "midpoints",
    "n_times",
    "bad_epochs",
    "bads",
    "n_epochs",
    "epoch_idx",
    "view_range",
    "xlim",
    "hscroll",
    "vline_pos",
    "vline_vis",
    "vline_label",
    "ovb_x_from_norm",
    "xmax",
    "epoch_dur",
    "decim",
    "scale_factor",
    "butterfly",
    "n_channels",
    "ch_start",
    "first_time",
    "n_epoch_traces",
    "vline_visible",
)


def load(path, backend):
    want = {f"base_{backend}.json", f"branch_{backend}.json"}
    stems = [os.path.join(path, f) for f in os.listdir(path) if f in want]
    assert len(stems) == 1, (stems, sorted(want))
    with open(stems[0]) as fid:
        blob = json.load(fid)
    return blob, np.load(stems[0][:-5] + ".npz")


def unhex(v):
    return float.fromhex(v) if isinstance(v, str) and v.startswith(("0x", "-0x")) else v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("branch")
    p.add_argument("--backend", default="matplotlib")
    p.add_argument("--field", default=None, help="dump every distinct value pair")
    a = p.parse_args()

    ab, an = load(a.base, a.backend)
    bb, bn = load(a.branch, a.backend)
    ar = {(r["case"], r["step"]): r for r in ab["records"]}
    br = {(r["case"], r["step"]): r for r in bb["records"]}

    # per-field: set of specs, max abs diff, max rel diff, distinct value pairs
    stats = defaultdict(
        lambda: dict(specs=set(), cases=set(), n=0, absmax=0.0, relmax=0.0, pairs={})
    )
    for k in sorted(set(ar) & set(br)):
        ra, rb = ar[k], br[k]
        sa, sb = ra.get("state"), rb.get("state")
        if sa is None or sb is None:
            continue
        spec = k[0].split("|")[0]
        for f in sorted(set(sa) | set(sb)):
            va, vb = sa.get(f), sb.get(f)
            if f in ("trace_x", "trace_y"):
                if va == vb:
                    continue
                if not isinstance(va, list) or not isinstance(vb, list):
                    continue
                bad = len(va) != len(vb) or any(
                    (x is None) != (y is None)
                    or (x is not None and x["sha"] != y["sha"])
                    for x, y in zip(va, vb)
                )
                if not bad:
                    continue
                s = stats[f]
                s["specs"].add(spec)
                s["cases"].add(k[0])
                s["n"] += 1
                continue
            if va == vb:
                continue
            s = stats[f]
            s["specs"].add(spec)
            s["cases"].add(k[0])
            s["n"] += 1
            if isinstance(va, dict) and isinstance(vb, dict):
                if (
                    va.get("npz") in an.files
                    and vb.get("npz") in bn.files
                    and va["shape"] == vb["shape"]
                ):
                    x, y = an[va["npz"]], bn[vb["npz"]]
                    d = np.abs(np.nan_to_num(x) - np.nan_to_num(y))
                    if d.size:
                        s["absmax"] = max(s["absmax"], float(d.max()))
                        denom = np.maximum(np.abs(np.nan_to_num(x)), 1e-300)
                        s["relmax"] = max(s["relmax"], float((d / denom).max()))
                continue
            fa, fb = unhex(va), unhex(vb)
            if isinstance(fa, float) and isinstance(fb, float):
                d = abs(fa - fb)
                s["absmax"] = max(s["absmax"], d)
                if fa:
                    s["relmax"] = max(s["relmax"], d / abs(fa))
            key = (repr(va), repr(vb))
            s["pairs"][key] = s["pairs"].get(key, 0) + 1

    print(f"backend={a.backend}  states compared={len(set(ar) & set(br))}")
    print(f"records only in base  : {sorted(set(ar) - set(br))[:5]} ...")
    print(f"records only in branch: {len(set(br) - set(ar))} states")
    print()
    print(f"{'field':22} {'n':>6} {'specs':>6} {'absmax':>12} {'relmax':>10}  critical")
    for f, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"{f:22} {s['n']:6d} {len(s['specs']):6d} "
            f"{s['absmax']:12.4g} {s['relmax']:10.3g}  {'YES' if f in CRITICAL else ''}"
        )
        print(f"    specs: {', '.join(sorted(s['specs']))}")
    if a.field:
        s = stats[a.field]
        print(f"\ndistinct value pairs for {a.field} (base -> branch):")
        for (x, y), n in sorted(s["pairs"].items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}x  {x}  ->  {y}")

    # untouched-fields census: prove the critical set really is clean
    clean = [f for f in CRITICAL if f not in stats]
    print(f"\nCRITICAL fields with zero divergence ({len(clean)}): {', '.join(clean)}")
    dirty = [f for f in CRITICAL if f in stats]
    print(f"CRITICAL fields that diverged ({len(dirty)}): {', '.join(dirty) or 'none'}")


if __name__ == "__main__":
    main()
