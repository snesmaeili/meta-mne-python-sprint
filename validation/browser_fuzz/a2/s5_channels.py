"""Channel configuration: ``picks``, ``order``, ``group_by``, bads, paging,
clicking a channel name, ``butterfly`` and ``epoch_colors``.

Every cell runs ragged and equal-duration and prints both, so the triage rule is
applied in the same output.
"""

import traceback

import numpy as np
from matplotlib.colors import to_rgba

import mne

from .. import invariants
from ..build import Spec, build
from . import display_checks
from .common import assert_env, close_fig, open_browser


def _drawn_names(fig):
    return [t.get_text() for t in fig.mne.ax_main.get_yticklabels()]


def _state(fig):
    return dict(
        picks=list(map(int, fig.mne.picks)),
        n_channels=int(fig.mne.n_channels),
        ch_start=int(fig.mne.ch_start),
        butterfly=bool(fig.mne.butterfly),
        labels=_drawn_names(fig),
        bads=list(fig.mne.info["bads"]),
        n_traces=len(fig.mne.traces),
        n_epoch_traces=len(fig.mne.epoch_traces),
    )


def run(spec, label, script=(), *, verbose=True, **plot_kwargs):
    """Open, replay ``script``, return (violations, states, error)."""
    viol, states, err = [], [], ""
    fig = None
    try:
        case = build(spec)
        fig = open_browser(case, "matplotlib", **plot_kwargs)

        def _check(tag):
            msgs = invariants.check_all(fig, case, "matplotlib")
            msgs += display_checks.check(
                fig, case, with_events=bool(plot_kwargs.get("events"))
            )
            viol.extend(f"[{tag}] {m}" for m in msgs)
            states.append((tag, _state(fig)))

        _check("open")
        for step in script:
            if isinstance(step, str):
                fig._fake_keypress(step)
                _check(f"key:{step}")
            else:
                step(fig, case)
                _check(getattr(step, "tag", "call"))
    except Exception:
        err = traceback.format_exc(limit=8)
    finally:
        close_fig(fig)
    if verbose:
        print(f"\n--- {label}")
        if err:
            print("    ERROR:\n" + "\n".join("      " + ln for ln in err.splitlines()))
        for tag, st in states:
            print(
                f"    {tag:<14} picks={st['picks'][:8]}{'...' if len(st['picks']) > 8 else ''} "
                f"ch_start={st['ch_start']} n_ch={st['n_channels']} "
                f"butterfly={st['butterfly']} traces={st['n_traces']} "
                f"labels={st['labels'][:6]}"
            )
        for m in viol:
            print(f"    VIOLATION {m}")
    return viol, states, err


def click_ch_name(which):
    def step(fig, case):
        labels = fig.mne.ax_main.get_yticklabels()
        if not labels:
            return
        lab = labels[which % len(labels)]
        fig._fake_click((0.5, 0.5), ax=fig.mne.ax_main, xform="ax")  # warm up
        # the browser binds bad-channel toggling to a pick on the tick label
        fig._toggle_bad_channel(which % len(labels))

    step.tag = f"toggle_ch:{which}"
    return step


def main():
    assert_env()
    mne.set_log_level("error")

    pairs = [
        ("ch1", Spec(lengths=(100, 250, 75, 180), n_channels=1),
         Spec(lengths=(100, 100, 100, 100), n_channels=1, force_fixed=True)),
        ("ch2", Spec(lengths=(100, 250, 75, 180), n_channels=2),
         Spec(lengths=(100, 100, 100, 100), n_channels=2, force_fixed=True)),
        ("ch64", Spec(lengths=(100, 250, 75, 180), n_channels=64),
         Spec(lengths=(100, 100, 100, 100), n_channels=64, force_fixed=True)),
        ("mixed16_bads3",
         Spec(lengths=(100, 250, 75, 180), n_channels=16, mixed_types=True, n_bads=3),
         Spec(lengths=(100, 100, 100, 100), n_channels=16, mixed_types=True,
              n_bads=3, force_fixed=True)),
    ]

    print("=" * 70)
    print("PAGING (pageup / pagedown) and channel scroll")
    for name, rag, fix in pairs:
        script = ["pagedown", "pagedown", "down", "down", "pageup", "up", "up"]
        run(rag, f"{name} RAGGED", script, n_epochs=2)
        run(fix, f"{name} FIXED ", script, n_epochs=2)

    print("\n" + "=" * 70)
    print("BUTTERFLY, including toggling mid-scroll")
    for name, rag, fix in pairs[2:]:
        script = ["right", "right", "b", "right", "b", "left", "b"]
        run(rag, f"{name} RAGGED butterfly", script, n_epochs=1)
        run(fix, f"{name} FIXED  butterfly", script, n_epochs=1)
    # butterfly from the start, on a 100:1 spread, scrolled into the middle
    run(Spec(lengths=(3, 300, 3, 300), n_channels=16, mixed_types=True),
        "spread100to1 RAGGED butterfly=True", ["right", "b", "right", "b"],
        n_epochs=1, butterfly=True)
    run(Spec(lengths=(300, 300, 300, 300), n_channels=16, mixed_types=True,
             force_fixed=True),
        "spread100to1 FIXED  butterfly=True", ["right", "b", "right", "b"],
        n_epochs=1, butterfly=True)

    print("\n" + "=" * 70)
    print("TOGGLING A CHANNEL BAD")
    for name, rag, fix in pairs[2:]:
        script = [click_ch_name(0), click_ch_name(3), "b", click_ch_name(1)]
        run(rag, f"{name} RAGGED bad-toggle", script, n_epochs=2)
        run(fix, f"{name} FIXED  bad-toggle", script, n_epochs=2)

    print("\n" + "=" * 70)
    print("PICKS / ORDER / GROUP_BY")
    rag = Spec(lengths=(100, 250, 75, 180), n_channels=16, mixed_types=True)
    fix = Spec(lengths=(100, 100, 100, 100), n_channels=16, mixed_types=True,
               force_fixed=True)
    for kw, lbl in [
        (dict(picks=["EEG000", "MAG003", "GRAD004"]), "picks=names"),
        (dict(picks="eeg"), "picks=eeg"),
        (dict(picks=[0]), "picks=[0]"),
        (dict(order=[5, 4, 3, 2, 1, 0]), "order=reversed"),
        (dict(group_by="original"), "group_by=original"),
        (dict(group_by="type"), "group_by=type"),
    ]:
        run(rag, f"RAGGED {lbl}", ["down", "b", "pagedown"], n_epochs=2, **kw)
        run(fix, f"FIXED  {lbl}", ["down", "b", "pagedown"], n_epochs=2, **kw)

    print("\n" + "=" * 70)
    print("GROUP_BY selection / position (needs a montage)")
    for gb in ("selection", "position"):
        for tag, spec in (("RAGGED", _montaged(ragged=True)),
                          ("FIXED ", _montaged(ragged=False))):
            _run_montaged(spec, f"{tag} group_by={gb}", group_by=gb)

    print("\n" + "=" * 70)
    print("EPOCH_COLORS")
    epoch_colors_probe()


def _montaged(ragged):
    return ("montage", ragged)


def _run_montaged(spec, label, **plot_kwargs):
    """Build 64 real EEG channel names so ``_setup_channel_selections`` works."""
    _, ragged = spec
    sfreq = 100.0
    montage = mne.channels.make_standard_montage("standard_1020")
    names = [n for n in montage.ch_names if n in
             mne.channels.read_layout("EEG1005").names][:64]
    if len(names) < 64:
        names = montage.ch_names[:64]
    rng = np.random.default_rng(0)
    info = mne.create_info(names, sfreq, ["eeg"] * len(names))
    info.set_montage(montage, on_missing="ignore")
    lengths = (100, 250, 75, 180) if ragged else (100, 100, 100, 100)
    stride = max(lengths) + int(sfreq) + 100
    events = np.column_stack([np.arange(4) * stride + stride,
                              np.zeros(4, int), np.ones(4, int)])
    try:
        if ragged:
            arrays = [rng.standard_normal((len(names), n)) * 1e-6 for n in lengths]
            ep = mne.EpochsArray(arrays, info, events=events, tmin=np.zeros(4),
                                 event_id={"x": 1}, baseline=None, verbose=False)
        else:
            data = rng.standard_normal((4, len(names), lengths[0])) * 1e-6
            ep = mne.EpochsArray(data, info, events=events, tmin=0.0,
                                 event_id={"x": 1}, baseline=None, verbose=False)
        mne.viz.set_browser_backend("matplotlib")
        fig = ep.plot(n_epochs=2, show=False, **plot_kwargs)
        fig.test_mode = True
        sels = list(fig.mne.ch_selections) if fig.mne.ch_selections else None
        print(f"\n--- {label}: selections={sels}")
        print(f"    picks={list(map(int, fig.mne.picks))[:8]} n_channels={fig.mne.n_channels}")
        for key in ("pagedown", "b", "right", "b", "pageup"):
            fig._fake_keypress(key)
            print(f"    after {key:<9} picks={list(map(int, fig.mne.picks))[:8]} "
                  f"n_ch={fig.mne.n_channels} butterfly={fig.mne.butterfly} "
                  f"traces={len(fig.mne.traces)}")
        close_fig(fig)
    except Exception:
        print(f"\n--- {label}: ERROR")
        traceback.print_exc(limit=8)


def epoch_colors_probe():
    """Each epoch drawn in its own colour: check the colours land on the right
    x ranges."""
    for tag, spec in (
        ("RAGGED", Spec(lengths=(100, 250, 75, 180), n_channels=3)),
        ("FIXED ", Spec(lengths=(100, 100, 100, 100), n_channels=3,
                        force_fixed=True)),
    ):
        case = build(spec)
        palette = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
        colors = [[palette[k]] * 3 for k in range(4)]
        try:
            fig = open_browser(case, "matplotlib", n_epochs=4, epoch_colors=colors)
        except Exception:
            print(f"\n--- {tag} epoch_colors: ERROR")
            traceback.print_exc(limit=6)
            continue
        segs = [fig.mne.traces[0]] + [
            t for t in fig.mne.epoch_traces
            if t.axes is fig.mne.ax_main
        ]
        print(f"\n--- {tag} epoch_colors: {len(segs)} coloured segments for trace 0")
        bad = []
        for ln in segs:
            x = np.ma.masked_array(ln.get_xdata())
            unmasked = np.ma.compressed(x)
            col = to_rgba(ln.get_color())
            if not len(unmasked):
                continue
            k_lo = int(np.searchsorted(case.boundary_times[1:], unmasked.min(),
                                       side="right"))
            k_hi = int(np.searchsorted(case.boundary_times[1:], unmasked.max(),
                                       side="right"))
            want = to_rgba(palette[min(k_lo, 3)])
            ok = k_lo == k_hi and np.allclose(col, want, atol=1e-6)
            print(f"    x=[{unmasked.min():.3f}, {unmasked.max():.3f}] "
                  f"epochs {k_lo}..{k_hi} colour={tuple(round(c, 3) for c in col)} "
                  f"want={tuple(round(c, 3) for c in want)} {'ok' if ok else '<-- MISMATCH'}")
            if not ok:
                bad.append((k_lo, k_hi, col, want))
        # also toggle an epoch bad and re-check
        close_fig(fig)


if __name__ == "__main__":
    main()
