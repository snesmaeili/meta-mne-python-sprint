"""The A2 plot-argument x channel-configuration matrix, matplotlib.

Each cell opens a browser, checks the harness invariants plus the A2 display
checks, replays a short display-key script, and re-checks. Every ragged cell has
an equal-duration control built from the same generator so the triage rule can
be applied in the same run.
"""

import sys
import traceback

import numpy as np

from .. import invariants
from ..build import Spec
from . import display_checks
from .common import assert_env, build, close_fig, open_browser

SCRIPTS = {
    "open_only": [],
    "butterfly": ["b"],
    "butterfly_scroll": ["b", "right", "right", "b"],
    "scroll_butterfly": ["right", "b", "right", "b"],
    "paging": ["pagedown", "pagedown", "pageup", "pageup"],
    "channels": ["down", "down", "up", "up"],
    "scale": ["+", "+", "-", "-"],
    "display": ["b", "d", "s", "0", "t", "b"],
    "home_end": ["end", "end", "home", "home"],
}


def cell(spec, plot_kwargs, script, *, label=""):
    """Return ``(violations, error)`` for one matrix cell."""
    viol = []
    err = ""
    fig = None
    try:
        case = build(spec)
        fig = open_browser(case, "matplotlib", **plot_kwargs)
        with_events = bool(plot_kwargs.get("events"))

        def _check(tag):
            msgs = invariants.check_all(fig, case, "matplotlib")
            msgs += display_checks.check(fig, case, with_events=with_events)
            viol.extend(f"[{tag}] {m}" for m in msgs)

        _check("open")
        for key in script:
            fig._fake_keypress(key)
            _check(f"key:{key}")
    except Exception:
        err = traceback.format_exc(limit=8)
    finally:
        close_fig(fig)
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
    return viol, err


# -- the matrix --------------------------------------------------------------

RAGGED = Spec(lengths=(100, 250, 75, 180), name="ref")
FIXED = Spec(lengths=(100, 100, 100, 100), force_fixed=True, name="ref_fixed")


def ch_specs():
    out = []
    for n in (1, 2, 3, 64, 306):
        out.append(
            (f"ch{n}", Spec(lengths=(100, 250, 75, 180), n_channels=n),
             Spec(lengths=(100, 100, 100, 100), n_channels=n, force_fixed=True))
        )
    out.append(
        ("mixed16", Spec(lengths=(100, 250, 75, 180), n_channels=16, mixed_types=True),
         Spec(lengths=(100, 100, 100, 100), n_channels=16, mixed_types=True,
              force_fixed=True))
    )
    out.append(
        ("mixed16_bads3",
         Spec(lengths=(100, 250, 75, 180), n_channels=16, mixed_types=True, n_bads=3),
         Spec(lengths=(100, 100, 100, 100), n_channels=16, mixed_types=True,
              n_bads=3, force_fixed=True))
    )
    out.append(
        ("spread100to1", Spec(lengths=(3, 300, 3, 300), n_channels=8),
         Spec(lengths=(300, 300, 300, 300), n_channels=8, force_fixed=True))
    )
    return out


KWARG_SETS = {
    "default": dict(n_epochs=2),
    "n_epochs_1": dict(n_epochs=1),
    "n_epochs_all": dict(n_epochs=4),
    "butterfly": dict(n_epochs=2, butterfly=True),
    "events": dict(n_epochs=2, events=True),
    "decim2": dict(n_epochs=2, decim=2),
    "decim4": dict(n_epochs=2, decim=4),
    "n_channels_2": dict(n_epochs=2, n_channels=2),
    "n_channels_1": dict(n_epochs=2, n_channels=1),
}


def main():
    assert_env()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    failures = 0
    total = 0
    for name, ragged, fixed in ch_specs():
        if only and only not in name:
            continue
        for kname, kw in KWARG_SETS.items():
            for sname, script in SCRIPTS.items():
                total += 1
                rv, re_ = cell(ragged, kw, script)
                fv, fe = cell(fixed, kw, script)
                if not (rv or re_ or fv or fe):
                    continue
                # only report what ragged does and fixed does not
                rset = {_strip(m) for m in rv}
                fset = {_strip(m) for m in fv}
                new = sorted(rset - fset)
                same = sorted(rset & fset)
                if re_ and not fe:
                    failures += 1
                    print(f"\n### {name} / {kname} / {sname}: RAGGED-ONLY ERROR")
                    print(re_)
                elif re_ and fe:
                    print(f"\n--- {name} / {kname} / {sname}: error on BOTH (pre-existing)")
                    print(f"    ragged: {re_.strip().splitlines()[-1]}")
                    print(f"    fixed:  {fe.strip().splitlines()[-1]}")
                elif fe and not re_:
                    print(f"\n--- {name} / {kname} / {sname}: FIXED-ONLY error")
                    print(fe)
                if new:
                    failures += 1
                    print(f"\n### {name} / {kname} / {sname}: RAGGED-ONLY violations")
                    for m in new[:8]:
                        print(f"    {m}")
                if same:
                    print(f"\n--- {name} / {kname} / {sname}: on BOTH paths (pre-existing)")
                    for m in same[:4]:
                        print(f"    {m}")
    print(f"\n{total} cells, {failures} ragged-only")


def _strip(msg):
    """Drop the numbers so a violation compares across the two paths."""
    import re

    return re.sub(r"[-+0-9][0-9eE.+-]*", "#", msg)


if __name__ == "__main__":
    main()
