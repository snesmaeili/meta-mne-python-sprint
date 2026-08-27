"""Clicking channel names to toggle bad, across channel counts and butterfly.

Uses the real pick path (``fig._click_ch_name``), the same one
``mne/viz/tests/test_epochs.py`` drives.
"""

import traceback

import numpy as np

import mne

from .. import invariants
from ..build import Spec, build
from . import display_checks
from .common import assert_env, close_fig, open_browser


def probe(spec, label, *, n_channels_shown=None, butterfly=False, picks=None,
          n_bads=0):
    kwargs = dict(n_epochs=2)
    if n_channels_shown:
        kwargs["n_channels"] = n_channels_shown
    if butterfly:
        kwargs["butterfly"] = True
    if picks is not None:
        kwargs["picks"] = picks
    viol = []
    try:
        case = build(spec)
        fig = open_browser(case, "matplotlib", **kwargs)
    except Exception:
        print(f"\n--- {label}: OPEN ERROR")
        traceback.print_exc(limit=6)
        return None
    print(f"\n--- {label}: {len(fig.mne.picks)} picks, bads={fig.mne.info['bads']}")

    def _check(tag):
        msgs = invariants.check_all(fig, case, "matplotlib")
        msgs += display_checks.check(fig, case)
        for m in msgs:
            viol.append(f"[{tag}] {m}")

    _check("open")
    n_shown = len(fig.mne.picks)
    for ci in (0, min(1, n_shown - 1), n_shown - 1):
        try:
            fig._click_ch_name(ch_index=ci, button=1)
        except Exception as exc:  # noqa: BLE001
            viol.append(f"[click_ch_name:{ci}] raised {type(exc).__name__}: {exc}")
            continue
        _check(f"click_ch_name:{ci}")
        names = fig.mne.ch_names[fig.mne.picks]
        colors = [tuple(np.round(t.get_color(), 3)) if not isinstance(t.get_color(), str)
                  else t.get_color()
                  for t in fig.mne.ax_main.get_yticklabels()]
        bad_mask = [n in fig.mne.info["bads"] for n in names]
        print(f"    after click {ci}: bads={fig.mne.info['bads']} "
              f"bad_mask={bad_mask[:6]} tick_colors={colors[:4]}")
        # a bad channel's trace must be drawn in ch_color_bad
        for ii, is_bad in enumerate(bad_mask):
            want = fig.mne.ch_color_bad if is_bad else None
            got = fig.mne.traces[ii].get_color()
            if want is not None:
                from matplotlib.colors import to_rgba

                if not np.allclose(to_rgba(got), to_rgba(want), atol=1e-6):
                    viol.append(
                        f"[click_ch_name:{ci}] trace {ii} ({names[ii]}) is bad but "
                        f"drawn {got}, want {want}"
                    )
    # scroll and re-check that the bad colouring survives
    for key in ("right", "b", "right", "b"):
        try:
            fig._fake_keypress(key)
        except Exception as exc:  # noqa: BLE001
            viol.append(f"[key:{key}] raised {type(exc).__name__}: {exc}")
            break
        _check(f"key:{key}")
    close_fig(fig)
    for m in viol:
        print(f"    VIOLATION {m}")
    return viol


def main():
    assert_env()
    mne.set_log_level("error")
    cases = []
    for n in (1, 2, 3, 64, 306):
        cases.append((f"ch{n}",
                      Spec(lengths=(100, 250, 75, 180), n_channels=n),
                      Spec(lengths=(100, 100, 100, 100), n_channels=n,
                           force_fixed=True)))
    cases.append((
        "mixed16_bads3",
        Spec(lengths=(100, 250, 75, 180), n_channels=16, mixed_types=True, n_bads=3),
        Spec(lengths=(100, 100, 100, 100), n_channels=16, mixed_types=True,
             n_bads=3, force_fixed=True),
    ))
    all_viol = 0
    for name, rag, fix in cases:
        rv = probe(rag, f"{name} RAGGED") or []
        fv = probe(fix, f"{name} FIXED ") or []
        rs = {_strip(m) for m in rv}
        fs = {_strip(m) for m in fv}
        new = sorted(rs - fs)
        if new:
            all_viol += len(new)
            print(f"\n### {name}: RAGGED-ONLY")
            for m in new:
                print(f"    {m}")
    # picks + butterfly corners
    probe(Spec(lengths=(100, 250, 75, 180), n_channels=16, mixed_types=True),
          "mixed16 RAGGED picks=[0,5] butterfly", picks=[0, 5], butterfly=True)
    probe(Spec(lengths=(100, 100, 100, 100), n_channels=16, mixed_types=True,
               force_fixed=True),
          "mixed16 FIXED  picks=[0,5] butterfly", picks=[0, 5], butterfly=True)
    probe(Spec(lengths=(3, 300, 3, 300), n_channels=1),
          "one channel, 100:1 spread RAGGED")
    probe(Spec(lengths=(300, 300, 300, 300), n_channels=1, force_fixed=True),
          "one channel, 100:1 spread FIXED ")
    print(f"\n{all_viol} ragged-only violations")


def _strip(msg):
    import re

    return re.sub(r"[-+0-9][0-9eE.+-]*", "#", msg)


if __name__ == "__main__":
    main()
