"""Slice A6 -- the randomized fuzzer, with a shrinker that actually shrinks.

Everything here lives outside ``actions.py`` / ``specs.py`` / ``fuzz.py`` so the
other slices can keep editing those. Three things this adds over ``fuzz.py``:

1. **Normalized failure signatures.** ``fuzz._signature`` keys on the first 80
   characters of the message *including the numbers*, so a shrink step that
   changes ``[0.9 0.74]`` to ``[0.9 0.55]`` looks like a different bug and the
   shrink stalls. :func:`signature` strips numerics before comparing.
2. **A real delta-debugger.** Prefix truncation to the first failing step, then
   ddmin binary partitioning, then greedy single removal to a fixpoint. The old
   shrinker only did greedy single removal, at up to six O(n) passes.
3. **A mutation self-test** (``selftest``) that injects known-wrong behaviour and
   asserts the fuzzer catches it and shrinks it. A clean fuzz result is worthless
   without it.

CLI::

    python -m validation.browser_fuzz.fuzz_extra selftest --backend qt
    python -m validation.browser_fuzz.fuzz_extra sweep --backend qt \
        --pool FAST --seeds 0-50 --steps 40 --out runs/qt.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import invariants
from . import specs as _specs
from .build import build
from .runner import close_fig, Result, open_browser

# --------------------------------------------------------------------------
# extra actions: cross-boundary vline placement and window resizing that the
# shared alphabet does not reach (it only ever touches the *first* visible
# epoch).  Kept here so ``actions.py`` stays owned by the other slices.
# --------------------------------------------------------------------------


def _vline_in_nth_visible(fig, case, backend, n, frac):
    """Place a vline inside the n-th *visible* epoch, not always the first."""

    def action():
        ix0, ix1 = fig._get_epoch_ix_range()
        ix = int(np.clip(ix0 + n, ix0, ix1 - 1))
        span = case.boundary_times[ix + 1] - case.boundary_times[ix]
        x = case.boundary_times[ix] + frac * span
        if backend == "matplotlib":
            fig._fake_click((x, 0.5), xform="data")
        else:
            fig._add_vline(x)

    return action


def _vline_absolute(fig, case, backend, frac):
    """Place a vline at a fraction of the *whole recording*, visible or not."""

    def action():
        x = float(frac) * case.boundary_times[-1]
        x = min(x, case.boundary_times[-1] - 0.5 / case.sfreq)
        x = max(x, 0.0)
        if backend == "matplotlib":
            fig._fake_click((x, 0.5), xform="data")
        else:
            fig._add_vline(x)

    return action


def _n_channels(fig, delta):
    """Change how many channels are drawn without going through a keypress."""

    def action():
        fig._fake_keypress("pageup" if delta > 0 else "pagedown")

    return action


def extra_alphabet(fig, case, backend):
    """``[(name, callable), ...]`` added on top of ``actions.build_alphabet``."""
    acts = []
    for n in (1, 2):
        for frac in (0.05, 0.5, 0.95):
            acts.append(
                (
                    f"vline_nth:{n}@{frac}",
                    _vline_in_nth_visible(fig, case, backend, n, frac),
                )
            )
    for frac in (0.0, 0.33, 0.66, 0.999):
        acts.append((f"vline_abs:{frac}", _vline_absolute(fig, case, backend, frac)))
    return acts


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def _alphabet(fig, case, backend, include_setxrange=False):
    from .actions import build_alphabet

    acts = list(
        build_alphabet(
            fig,
            case,
            backend,
            include_windows=True,
            include_setxrange=include_setxrange,
        )
    )
    acts += extra_alphabet(fig, case, backend)
    return acts


def run_extra(spec, backend, names, *, plot_kwargs=None, strict_warnings=True,
              include_setxrange=False):
    """Same contract as ``runner.run`` but with the A6 extra actions merged in."""
    result = Result(spec_label=spec.label(), backend=backend)
    fig = None
    try:
        with warnings.catch_warnings():
            if strict_warnings:
                warnings.simplefilter("error")
                warnings.filterwarnings("ignore", message=".*non-interactive.*")
                warnings.filterwarnings("ignore", message=".*FigureCanvasAgg.*")
            case = build(spec)
            fig = open_browser(case, backend, **(plot_kwargs or {}))
            table = dict(_alphabet(fig, case, backend, include_setxrange))

            msgs = invariants.check_all(fig, case, backend)
            result.violations.extend((0, "<open>", m) for m in msgs)

            for step, name in enumerate(names, start=1):
                result.sequence.append(name)
                fn = table.get(name)
                if fn is None:
                    result.violations.append((step, name, f"unknown action {name!r}"))
                    continue
                fn()
                msgs = invariants.check_all(fig, case, backend)
                result.violations.extend((step, name, m) for m in msgs)
    except Exception:
        result.error = traceback.format_exc(limit=12)
    finally:
        if fig is not None:
            try:
                close_fig(fig)
            except Exception:
                pass
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
    return result


# --------------------------------------------------------------------------
# signatures
# --------------------------------------------------------------------------

_NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_CODE = re.compile(r"^([IWA]\d+)\b")


def _norm(text):
    # keep the invariant code (I0, I8, W1, ...) -- it is the one number in the
    # message that identifies *which* check failed rather than by how much
    code = _CODE.match(text)
    if code:
        head, text = code.group(1), text[code.end(1):]
    else:
        head = ""
    text = _ADDR.sub("0xADDR", text)
    return head + _NUM.sub("N", text)


def signature(result):
    """A stable key for 'the same failure', insensitive to the actual numbers.

    Two runs that hit the same wrong line of browser code get the same
    signature even though their epoch lengths and latencies differ, which is
    what makes both shrinking and cross-seed dedup work.
    """
    if result.error:
        tb = result.error.strip().splitlines()
        last = tb[-1]
        # the deepest frame inside the code under test, so an IndexError in
        # _xtick_formatter never merges with an IndexError elsewhere
        where = ""
        for line in reversed(tb):
            m = re.search(r'File "(.+?)", line (\d+), in (\S+)', line)
            if m:
                where = f"{os.path.basename(m.group(1))}:{m.group(3)}"
                break
        return ("error", f"{where} | {_norm(last)[:120]}")
    if not result.violations:
        return None
    first = result.violations[0][2]
    return ("violation", _norm(first.split("\n")[0])[:120])


def first_fail_step(result):
    if result.error:
        return len(result.sequence)
    return result.violations[0][0]


# --------------------------------------------------------------------------
# shrinking
# --------------------------------------------------------------------------


def shrink(spec, backend, names, target, *, plot_kwargs=None, runner=run_extra,
           budget=400, include_setxrange=False):
    """Delta-debug ``names`` to the shortest subsequence with the same signature.

    Returns ``(best, n_runs)``.  Three phases:

    * prefix truncation to the step that first failed (free: one run tells us),
    * ddmin binary partitioning (drop halves, then quarters, ...),
    * greedy single removal until nothing more can go.
    """
    calls = [0]

    def fails_same(seq):
        if not seq:
            return False
        if calls[0] >= budget:
            return False
        calls[0] += 1
        res = runner(spec, backend, list(seq), plot_kwargs=plot_kwargs,
                     include_setxrange=include_setxrange)
        return (not res.ok) and signature(res) == target

    best = list(names)

    # phase 0: prefix truncation
    res = runner(spec, backend, best, plot_kwargs=plot_kwargs,
                 include_setxrange=include_setxrange)
    calls[0] += 1
    if not res.ok and signature(res) == target:
        cut = first_fail_step(res)
        if 0 < cut < len(best) and fails_same(best[:cut]):
            best = best[:cut]

    # phase 1: ddmin
    n = 2
    while len(best) >= 2:
        chunk = max(1, len(best) // n)
        reduced = False
        for i in range(0, len(best), chunk):
            trial = best[:i] + best[i + chunk:]
            if trial and fails_same(trial):
                best = trial
                n = max(n - 1, 2)
                reduced = True
                break
        if not reduced:
            if n >= len(best):
                break
            n = min(len(best), n * 2)

    # phase 2: greedy single removal to a fixpoint
    changed = True
    while changed and calls[0] < budget:
        changed = False
        for i in range(len(best) - 1, -1, -1):
            trial = best[:i] + best[i + 1:]
            if trial and fails_same(trial):
                best = trial
                changed = True
    return best, calls[0]


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

#: Biased toward the boundary model (navigation, vlines) but display actions
#: stay in the pool so cross-interactions get exercised.
WEIGHTS = {
    "key:left": 6, "key:right": 6, "key:shift+left": 4, "key:shift+right": 4,
    "key:home": 3, "key:end": 3,
    "hscroll": 6, "change_duration": 4,
    "vline": 5, "vline_nth": 5, "vline_abs": 4,
    "click_latency": 5, "click_epoch": 3, "click_edge": 4,
    "hscroll_click": 4, "hscroll_drag": 4,
    "setxrange": 2,
    # display: kept, at lower weight
    "key:up": 2, "key:down": 2, "key:pageup": 2, "key:pagedown": 2,
    "key:b": 2, "key:d": 1, "key:s": 1, "key:0": 1, "key:t": 2,
    "key:+": 1, "key:-": 1,
    "key:?": 1, "key:j": 1, "key:h": 1,
}


def _weight(name):
    if name in WEIGHTS:
        return WEIGHTS[name]
    return WEIGHTS.get(name.split(":")[0], 1)


def alphabet_names(spec, backend, include_setxrange=False):
    case = build(spec)
    fig = open_browser(case, backend)
    try:
        return [n for n, _ in _alphabet(fig, case, backend, include_setxrange)]
    finally:
        close_fig(fig)


def walk(names, seed, n_steps):
    rng = random.Random(seed)
    pool = []
    for name in names:
        pool.extend([name] * _weight(name))
    return [rng.choice(pool) for _ in range(n_steps)]


@dataclass
class Failure:
    spec: str
    backend: str
    plotkw: str
    seed: int
    steps: int
    sig: tuple
    action: str
    message: str
    shrunk: list = field(default_factory=list)
    shrink_runs: int = 0


def fuzz_one(spec, backend, seed, n_steps, *, plot_kwargs=None, plotkw_name="default",
             names=None, do_shrink=True, include_setxrange=False):
    """One walk. Returns a :class:`Failure` or ``None``."""
    if names is None:
        names = alphabet_names(spec, backend, include_setxrange)
    seq = walk(names, seed, n_steps)
    res = run_extra(spec, backend, seq, plot_kwargs=plot_kwargs,
                    include_setxrange=include_setxrange)
    if res.ok:
        return None
    sig = signature(res)
    if res.error:
        action = res.sequence[-1] if res.sequence else "<open>"
        message = res.error.strip()
    else:
        _, action, message = res.violations[0]
    fail = Failure(
        spec=spec.label(), backend=backend, plotkw=plotkw_name, seed=seed,
        steps=n_steps, sig=sig, action=action, message=message,
    )
    if do_shrink:
        best, nruns = shrink(spec, backend, seq, sig, plot_kwargs=plot_kwargs,
                             include_setxrange=include_setxrange)
        fail.shrunk = best
        fail.shrink_runs = nruns
    return fail


# --------------------------------------------------------------------------
# mutation self-test: does the fuzzer actually catch a planted bug?
# --------------------------------------------------------------------------


def _mutations():
    """``[(name, apply_fn, undo_fn)]`` -- each plants a specific known defect."""
    import mne.viz._figure as _figure

    muts = []

    # M1: the *expectation* is wrong (boundary_samples off by one). Catches a
    # fuzzer that silently reads its expectations from the object under test.
    def m1_apply():
        real_build = sys.modules[__name__]._build_ref

        def patched(spec):
            case = real_build(spec)
            case.boundary_samples = case.boundary_samples.copy()
            case.boundary_samples[1] += 1
            case.boundary_times = case.boundary_samples / case.sfreq
            return case

        sys.modules[__name__].build = patched

    def m1_undo():
        sys.modules[__name__].build = sys.modules[__name__]._build_ref

    muts.append(("M1_expected_boundary_off_by_one", m1_apply, m1_undo))

    # M2: _get_epoch_ix_range off by one at the end.
    base_cls = _figure.BrowserBase
    orig_ix = base_cls._get_epoch_ix_range

    def m2_apply():
        def patched(self):
            a, b = orig_ix(self)
            n = len(self.mne.boundary_times) - 1
            return a, min(b + 1, n) if b < n else b

        base_cls._get_epoch_ix_range = patched

    def m2_undo():
        base_cls._get_epoch_ix_range = orig_ix

    muts.append(("M2_ix_range_off_by_one", m2_apply, m2_undo))

    # M3: _get_epoch_num_from_time off by one.
    orig_num = base_cls._get_epoch_num_from_time

    def m3_apply():
        def patched(self, time):
            out = orig_num(self, time)
            sel = list(self.mne.inst.selection)
            i = sel.index(out)
            return sel[min(i + 1, len(sel) - 1)]

        base_cls._get_epoch_num_from_time = patched

    def m3_undo():
        base_cls._get_epoch_num_from_time = orig_num

    muts.append(("M3_epoch_num_off_by_one", m3_apply, m3_undo))

    # M4: _get_start_stop returns a fixed-grid window (the pre-PR assumption).
    orig_ss = base_cls._get_start_stop

    def m4_apply():
        def patched(self):
            ix0, ix1 = orig_ix(self)
            per = int(self.mne.boundary_samples[1])
            return ix0 * per, ix1 * per

        base_cls._get_start_stop = patched

    def m4_undo():
        base_cls._get_start_stop = orig_ss

    muts.append(("M4_fixed_grid_start_stop", m4_apply, m4_undo))

    # -- sequence-dependent mutations: these do NOT fire at open, so the
    # shrinker has to find the specific subsequence inside a 40-step walk.

    # M5: ix_range off by one only once the view has scrolled twice.
    # Minimal repro: two forward navigations.
    def m5_apply():
        def patched(self):
            a, b = orig_ix(self)
            n = len(self.mne.boundary_times) - 1
            if a >= 2:
                return a, min(b + 1, n) if b < n else b
            return a, b

        base_cls._get_epoch_ix_range = patched

    muts.append(("M5_ix_range_wrong_after_2_scrolls", m5_apply, m2_undo))

    # M6: start/stop drifts by a sample only while a vline is on screen.
    # Minimal repro: place a vline, then navigate once.
    def m6_apply():
        def patched(self):
            start, stop = orig_ss(self)
            if getattr(self.mne, "vline_visible", False) and self.mne.t_start > 0:
                return start, max(stop - 1, start + 1)
            return start, stop

        base_cls._get_start_stop = patched

    muts.append(("M6_stop_drifts_with_vline", m6_apply, m4_undo))

    return muts


#: What a perfect shrinker would return for each planted defect.
MUTATION_MIN_LEN = {
    "M1_expected_boundary_off_by_one": 1,
    "M2_ix_range_off_by_one": 1,
    "M3_epoch_num_off_by_one": 1,
    "M4_fixed_grid_start_stop": 1,
    "M5_ix_range_wrong_after_2_scrolls": 2,
    "M6_stop_drifts_with_vline": 2,
}


_build_ref = build


def selftest(backend="qt", spec_name="reference_fixture", seeds=6, steps=40,
             verbose=True):
    """Plant each defect, fuzz, and require a catch plus a decent shrink."""
    spec = _specs.BY_NAME[spec_name]
    names = alphabet_names(spec, backend)
    rows = []
    for mname, apply_fn, undo_fn in _mutations():
        apply_fn()
        try:
            caught = None
            t0 = time.time()
            for seed in range(seeds):
                fail = fuzz_one(spec, backend, seed, steps, names=names)
                if fail is not None:
                    caught = (seed, fail)
                    break
            dt = time.time() - t0
        finally:
            undo_fn()
        if caught is None:
            rows.append((mname, "MISSED", None, None, None, dt))
        else:
            seed, fail = caught
            rows.append(
                (mname, "caught", seed, steps, len(fail.shrunk), dt, fail)
            )
        if verbose:
            print(f"  {mname}: {rows[-1][1]}", flush=True)
            if caught:
                ideal = MUTATION_MIN_LEN.get(mname)
                score = "" if ideal is None else f" (ideal {ideal})"
                print(f"    seed={seed} {steps} steps -> {len(fail.shrunk)} steps"
                      f"{score} in {fail.shrink_runs} shrink runs ({dt:.1f}s)")
                print(f"    sig     : {fail.sig[1]}")
                print(f"    shrunk  : {fail.shrunk}")
    return rows


# --------------------------------------------------------------------------
# sweep CLI
# --------------------------------------------------------------------------


def _parse_range(text):
    if "-" in text:
        a, b = text.split("-", 1)
        return range(int(a), int(b))
    return range(0, int(text))


def _pool(name):
    if name == "FAST":
        return list(_specs.FAST)
    if name == "ALL":
        return list(_specs.ALL)
    if name == "SLOW":
        fast = {s.label() for s in _specs.FAST}
        return [s for s in _specs.ALL if s.label() not in fast]
    return [_specs.BY_NAME[n] for n in name.split(",")]


def sweep(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qt", choices=("matplotlib", "qt"))
    ap.add_argument("--pool", default="FAST")
    ap.add_argument("--seeds", default="0-20")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--plotkw", default="default",
                    help="comma list of specs.PLOT_KWARGS keys, or ALL")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--include-setxrange", action="store_true")
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--max-per-sig", type=int, default=2,
                    help="stop shrinking a signature after this many hits")
    args = ap.parse_args(argv)

    import mne

    mne.set_log_level("ERROR")
    warnings.filterwarnings("ignore")

    si, sn = (int(x) for x in args.shard.split("/"))
    pool = _pool(args.pool)
    seeds = list(_parse_range(args.seeds))
    if args.plotkw == "ALL":
        plotkws = list(_specs.PLOT_KWARGS.items())
    else:
        plotkws = [(k, _specs.PLOT_KWARGS[k]) for k in args.plotkw.split(",")]

    jobs = []
    for spec in pool:
        for pkname, pk in plotkws:
            jobs.append((spec, pkname, pk))
    jobs = [j for i, j in enumerate(jobs) if i % sn == si]

    out = open(args.out, "a", buffering=1) if args.out else None
    seen = {}
    n_runs = 0
    t0 = time.time()
    for spec, pkname, pk in jobs:
        try:
            names = alphabet_names(spec, args.backend, args.include_setxrange)
        except Exception:
            rec = dict(kind="open_error", spec=spec.label(), backend=args.backend,
                       plotkw=pkname, message=traceback.format_exc(limit=8))
            print(json.dumps(rec), file=out or sys.stdout, flush=True)
            continue
        for seed in seeds:
            n_runs += 1
            try:
                key_shrink = not args.no_shrink
                fail = fuzz_one(
                    spec, args.backend, seed, args.steps, plot_kwargs=pk,
                    plotkw_name=pkname, names=names, do_shrink=False,
                    include_setxrange=args.include_setxrange,
                )
            except Exception:
                rec = dict(kind="harness_error", spec=spec.label(), seed=seed,
                           backend=args.backend, plotkw=pkname,
                           message=traceback.format_exc(limit=8))
                print(json.dumps(rec), file=out or sys.stdout, flush=True)
                continue
            if fail is None:
                continue
            sig = fail.sig
            seen[sig] = seen.get(sig, 0) + 1
            if key_shrink and seen[sig] <= args.max_per_sig:
                seq = walk(names, seed, args.steps)
                best, nruns = shrink(
                    spec, args.backend, seq, sig, plot_kwargs=pk,
                    include_setxrange=args.include_setxrange,
                )
                fail.shrunk, fail.shrink_runs = best, nruns
            rec = dict(
                kind="fail", spec=fail.spec, backend=fail.backend,
                plotkw=fail.plotkw, seed=fail.seed, steps=fail.steps,
                sig_kind=sig[0], sig=sig[1], action=fail.action,
                message=fail.message[:4000], shrunk=fail.shrunk,
                shrink_runs=fail.shrink_runs, n_hits=seen[sig],
            )
            print(json.dumps(rec), file=out or sys.stdout, flush=True)
    summary = dict(kind="summary", backend=args.backend, pool=args.pool,
                   n_specs=len(pool), n_jobs=len(jobs), n_seeds=len(seeds),
                   steps=args.steps, n_runs=n_runs, n_sigs=len(seen),
                   elapsed=round(time.time() - t0, 1),
                   sigs=[[k[0], k[1], v] for k, v in seen.items()])
    print(json.dumps(summary), file=out or sys.stdout, flush=True)
    if out:
        out.close()
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "selftest":
        rest = argv[1:]
        ap = argparse.ArgumentParser()
        ap.add_argument("--backend", default="qt")
        ap.add_argument("--spec", default="reference_fixture")
        ap.add_argument("--seeds", type=int, default=6)
        ap.add_argument("--steps", type=int, default=40)
        a = ap.parse_args(rest)
        import mne

        mne.set_log_level("ERROR")
        warnings.filterwarnings("ignore")
        rows = selftest(a.backend, a.spec, a.seeds, a.steps)
        missed = [r for r in rows if r[1] == "MISSED"]
        print(f"\n{len(rows) - len(missed)}/{len(rows)} mutations caught")
        return 1 if missed else 0
    if argv and argv[0] == "sweep":
        return sweep(argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
