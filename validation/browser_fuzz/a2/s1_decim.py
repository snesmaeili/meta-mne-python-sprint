"""S7 -- display decim phase across epoch boundaries (matplotlib)."""

import sys

import numpy as np

from ..build import Spec
from .common import (
    assert_env,
    close_fig,
    build,
    drawn_samples,
    open_browser,
    per_epoch_phase,
    trace_matches_source,
)


def probe(spec, decim, n_epochs):
    case = build(spec)
    fig = open_browser(case, "matplotlib", n_epochs=n_epochs, decim=decim)
    samples = drawn_samples(fig, case)
    phases = per_epoch_phase(samples, case)
    complaints = trace_matches_source(fig, case)
    close_fig(fig)
    return case, samples, phases, complaints


def show(title, spec, decim, n_epochs):
    case, samples, phases, complaints = probe(spec, decim, n_epochs)
    lens = list(case.lengths)
    print(f"\n=== {title}: lengths={lens} decim={decim} n_epochs={n_epochs}")
    print(f"    boundary_samples={list(case.boundary_samples)}")
    print(f"    drawn {len(samples)} samples, first={samples[0]} last={samples[-1]}")
    for k, (ph, n) in phases.items():
        expect0 = 0
        flag = "  <-- PHASE != 0" if ph != expect0 else ""
        print(f"    epoch {k}: first drawn within-epoch index {ph}, n={n}{flag}")
    if complaints:
        for c in complaints:
            print(f"    XY MISMATCH: {c}")
    else:
        print("    x/y pairing: source-consistent")
    return phases


def main():
    assert_env()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    # lengths deliberately coprime to 2, 4 and 8
    ragged = Spec(lengths=(101, 103, 107, 109), name="a2_ragged_coprime")
    ragged_even = Spec(lengths=(100, 104, 108, 112), name="a2_ragged_mult4")
    fixed_odd = Spec(
        lengths=(101, 101, 101, 101), force_fixed=True, name="a2_fixed_odd"
    )
    fixed_even = Spec(
        lengths=(100, 100, 100, 100), force_fixed=True, name="a2_fixed_even"
    )
    listpath_odd = Spec(lengths=(101, 101, 101, 101), name="a2_listpath_odd")

    for d in (1, 2, 4, 8):
        if which in ("all", "ragged"):
            show("RAGGED coprime", ragged, d, 4)
        if which in ("all", "ragged"):
            show("RAGGED multiples of 4", ragged_even, d, 4)
        if which in ("all", "fixed"):
            show("FIXED L=101 (L%d != 0)", fixed_odd, d, 4)
            show("FIXED L=100", fixed_even, d, 4)
            show("LIST-PATH equal L=101", listpath_odd, d, 4)


if __name__ == "__main__":
    main()
