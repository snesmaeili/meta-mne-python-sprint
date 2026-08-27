"""Random action walks with shrinking.

A failing seed is only useful if it reduces to something a person can read, so
every failure is delta-debugged down to the shortest prefix-and-subset that
still violates the same invariant.
"""

import argparse
import random

from . import specs as _specs
from .actions import DISPLAY_KEYS, NAV_KEYS
from .build import build
from .runner import open_browser, run


def alphabet_names(spec, backend):
    """Action names available for this spec and backend, without opening a run."""
    from .actions import build_alphabet

    case = build(spec)
    fig = open_browser(case, backend)
    try:
        return [name for name, _ in build_alphabet(fig, case, backend)]
    finally:
        fig.close()


def _signature(result):
    """What makes two failures 'the same', for shrinking."""
    if result.error:
        return ("error", result.error.strip().splitlines()[-1])
    first = result.violations[0][2]
    return ("violation", first.split("\n")[0][:80])


def shrink(spec, backend, names, target, plot_kwargs=None, max_passes=6):
    """Return the shortest subsequence still failing the same way."""
    best = list(names)
    for _ in range(max_passes):
        changed = False
        # try dropping one action at a time, from the end so prefixes win
        for i in range(len(best) - 1, -1, -1):
            trial = best[:i] + best[i + 1 :]
            if not trial:
                continue
            res = run(spec, backend, trial, plot_kwargs=plot_kwargs)
            if not res.ok and _signature(res) == target:
                best = trial
                changed = True
        if not changed:
            break
    return best


def fuzz_one(spec, backend, seed, n_steps, plot_kwargs=None, weights=None):
    """One random walk. Returns ``(result, shrunk_names)`` or ``(result, None)``."""
    names = alphabet_names(spec, backend)
    rng = random.Random(seed)
    if weights:
        pool = []
        for name in names:
            pool.extend([name] * weights.get(name.split(":")[0], 1))
    else:
        pool = names
    seq = [rng.choice(pool) for _ in range(n_steps)]
    result = run(spec, backend, seq, plot_kwargs=plot_kwargs)
    if result.ok:
        return result, None
    return result, shrink(spec, backend, seq, _signature(result), plot_kwargs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="matplotlib", choices=("matplotlib", "qt"))
    ap.add_argument("--spec", default=None, help="spec name; default sweeps FAST")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--stop-after", type=int, default=0, help="0 = never stop")
    args = ap.parse_args(argv)

    pool = [_specs.BY_NAME[args.spec]] if args.spec else _specs.FAST
    n_fail = 0
    for spec in pool:
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            result, shrunk = fuzz_one(spec, args.backend, seed, args.steps)
            if result.ok:
                continue
            n_fail += 1
            print(f"\n=== FAIL {spec.label()} [{args.backend}] seed={seed} ===")
            print(result.report())
            if shrunk is not None:
                print(f"  shrunk to: {shrunk}")
            if args.stop_after and n_fail >= args.stop_after:
                return 1
    print(f"\n{n_fail} failing seeds over {len(pool)} specs")
    return 1 if n_fail else 0


# Sensible bias: navigation is where the boundary model lives.
DEFAULT_WEIGHTS = {"key": 2, "click_epoch": 1, "click_latency": 2, "hscroll": 3,
                   "change_duration": 2, "setxrange": 2, "vline": 2}

NAV_ONLY = list(NAV_KEYS)
DISPLAY_ONLY = list(DISPLAY_KEYS)

if __name__ == "__main__":
    raise SystemExit(main())
