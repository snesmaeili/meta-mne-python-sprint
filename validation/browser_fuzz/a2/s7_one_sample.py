"""A2-1 / A2-2 repro: a 1-sample epoch is not drawn as itself.

Three symptoms of one cause -- ``epoch_ix`` in ``_draw_traces`` is rebuilt by
searchsorting the *time range* instead of asking the view which epochs it holds
(:mod:`mne.viz._mpl_figure`, line 2266) -- plus the colour mask at line 2337,
which starts each epoch's band one sample late.

Run with::

    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
      PYTHONPATH="D:/mne-python;D:/meta-mne-python-sprint" \
      python -m validation.browser_fuzz.a2.s7_one_sample
"""

import numpy as np
from matplotlib.colors import to_rgba

import mne

from ..build import Spec, build
from .common import assert_env, close_fig, open_browser

RED = to_rgba((1, 0, 0))


def bands(fig):
    out = []
    for ln in [fig.mne.traces[0]] + list(fig.mne.epoch_traces):
        x = np.ma.compressed(np.ma.masked_array(ln.get_xdata()))
        if len(x):
            out.append((x, to_rgba(ln.get_color())))
    return sorted(out, key=lambda g: g[0].min())


def report(spec, label, *, n_epochs=2, epoch_colors=None, bad=None, scrolls=0):
    case = build(spec)
    kwargs = dict(n_epochs=n_epochs)
    if epoch_colors is not None:
        kwargs["epoch_colors"] = epoch_colors
    print(f"\n--- {label}  lengths={list(case.lengths)}")
    try:
        fig = open_browser(case, "matplotlib", **kwargs)
        for _ in range(scrolls):
            fig._fake_keypress("right")
        if bad is not None:
            fig.mne.bad_epochs = [int(fig.mne.inst.selection[bad])]
            fig._redraw()
    except Exception as exc:  # noqa: BLE001
        print(f"    {type(exc).__name__}: {exc}")
        return
    ix0, ix1 = fig._get_epoch_ix_range()
    start, stop = fig._get_start_stop()
    tr = (fig.mne.times + fig.mne.first_time)[[0, -1]]
    ss = np.searchsorted(fig.mne.boundary_times, tr)
    print(f"    view epochs {ix0}:{ix1}, samples [{start}, {stop}); "
          f"time_range={np.round(tr, 4).tolist()} -> epoch_ix={list(range(ss[0], ss[1]))}")
    gs = bands(fig)
    painted = np.unique(np.concatenate([g[0] for g in gs])) if gs else np.array([])
    loaded = np.round(np.arange(start, stop) / case.sfreq, 9)
    print(f"    unpainted loaded samples: "
          f"{np.setdiff1d(loaded, np.round(painted, 9)).round(4).tolist()}")
    for x, c in gs:
        print(f"    band x[{x.min():.3f}..{x.max():.3f}] n={len(x)} "
              f"colour={tuple(round(v, 2) for v in c)}"
              f"{' RED' if np.allclose(c, RED, atol=1e-6) else ''}")
    for k in range(ix0, ix1):
        lo, hi = case.boundary_times[k], case.boundary_times[k + 1]
        n = sum(int(((x >= lo - 1e-12) & (x < hi - 1e-12)).sum()) for x, _ in gs)
        if n == 0:
            print(f"    *** epoch {k} ({case.lengths[k]} sample(s)) is in view "
                  f"and paints nothing")
    close_fig(fig)


def main():
    assert_env()
    mne.set_log_level("error")
    pal3 = [["#ff0000"], ["#00ff00"], ["#0000ff"]]

    print("=" * 70)
    print("A2-1  a 1-sample epoch marked bad is drawn in the good colour")
    report(Spec(lengths=(100, 1, 100), n_channels=1),
           "RAGGED (100,1,100), epoch 1 bad", bad=1)
    report(Spec(lengths=(100, 100, 100), n_channels=1, force_fixed=True),
           "FIXED  (100,)x3, epoch 1 bad  [control]", bad=1)
    report(Spec(lengths=(100, 1, 100), n_channels=1),
           "RAGGED (100,1,100), epoch_colors R/G/B", epoch_colors=pal3)
    report(Spec(lengths=(100, 100, 100), n_channels=1, force_fixed=True),
           "FIXED  (100,)x3, epoch_colors R/G/B  [control]", epoch_colors=pal3)

    print("\n" + "=" * 70)
    print("A2-2  a 1-sample epoch first in the view paints nothing")
    report(Spec(lengths=(1, 100, 100), n_channels=1), "RAGGED (1,100,100)")
    report(Spec(lengths=(100, 1, 100, 100), n_channels=1),
           "RAGGED (100,1,100,100) scrolled onto the 1-sample epoch", scrolls=1)

    print("\n" + "=" * 70)
    print("A2-3  n_epochs=1 on a 1-sample epoch raises IndexError")
    report(Spec(lengths=(100, 1, 100, 100), n_channels=1),
           "RAGGED (100,1,100,100) n_epochs=1, scrolled", n_epochs=1, scrolls=1)
    report(Spec(lengths=(1, 100, 100, 100), n_channels=1),
           "RAGGED (1,100,100,100) n_epochs=1", n_epochs=1)
    report(Spec(lengths=(100, 100, 100, 100), n_channels=1, force_fixed=True),
           "FIXED  (100,)x4 n_epochs=1  [control]", n_epochs=1)


if __name__ == "__main__":
    main()
