"""A7 parity guard: equal-duration epochs must browse identically pre/post PR.

This module is deliberately **standalone**. It must import and run against the
pre-PR worktrees (``D:/tmp/mne-base`` + ``D:/tmp/qtb-base/src``) as well as the
branch, so it never imports the rest of the fuzz harness, never touches
``_get_epoch_ix_range`` / ``_n_times_per_epoch`` / ``variable_duration``, and
builds only equal-duration epochs through the 3-D ``EpochsArray`` path.

Every expectation is a *recorded* state from the other environment; nothing here
asks the browser to confirm its own arithmetic.

Usage
-----
Record one environment::

    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
    PYTHONPATH="D:/tmp/mne-base;D:/tmp/qtb-base/src" \
    python parity.py record --backend matplotlib --out D:/tmp/parity/base

Diff two recordings::

    python parity.py diff D:/tmp/parity/base D:/tmp/parity/branch
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import traceback
import warnings
from dataclasses import dataclass, field

import numpy as np

warnings.simplefilter("ignore")

# --------------------------------------------------------------------------
# environment identification
# --------------------------------------------------------------------------


def flavour():
    """Return "branch" or "base", asserted from a feature only the PR has."""
    import mne

    has_vd = hasattr(mne.EpochsArray, "variable_duration")
    from mne.viz._figure import BrowserBase

    has_ix = hasattr(BrowserBase, "_get_epoch_ix_range")
    if has_vd and has_ix:
        return "branch"
    if not has_vd and not has_ix:
        return "base"
    raise RuntimeError(
        f"inconsistent environment: variable_duration={has_vd}, "
        f"_get_epoch_ix_range={has_ix}"
    )


def env_report():
    import mne
    import mne_qt_browser

    return dict(
        flavour=flavour(),
        mne_file=mne.__file__,
        mne_version=mne.__version__,
        qtb_file=mne_qt_browser.__file__,
        numpy=np.__version__,
        python=sys.version.split()[0],
    )


# --------------------------------------------------------------------------
# the equal-duration matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PSpec:
    name: str
    n_src: int  # epochs built
    n_samples: int  # samples per epoch (equal by construction)
    n_channels: int = 3
    sfreq: float = 100.0
    tmin: float = 0.0
    drop: tuple = ()

    @property
    def n_kept(self):
        return self.n_src - len(self.drop)


SPECS = [
    PSpec("e1_L137", 1, 137),
    PSpec("e2_L137", 2, 137),
    PSpec("e4_L100", 4, 100),
    PSpec("e5_L77_ch1", 5, 77, n_channels=1),
    PSpec("e50_L40", 50, 40),
    PSpec("e4_L100_ch64_sf250_tneg", 4, 100, n_channels=64, sfreq=250.0, tmin=-0.2),
    PSpec("e4_L250_sf250_tneg", 4, 250, sfreq=250.0, tmin=-0.2),
    PSpec("e4_L100_sf512p3_tneg", 4, 100, sfreq=512.3, tmin=-0.2),
    PSpec("e4_L100_sf512p3_tpos", 4, 100, sfreq=512.3, tmin=0.1),
    PSpec("e4_L200_sf1000_tpos", 4, 200, sfreq=1000.0, tmin=0.1),
    PSpec("e6_L100_drop_noncontig", 6, 100, drop=(1, 3)),
    PSpec("e5_L120_sf250_drop_first", 5, 120, sfreq=250.0, tmin=-0.2, drop=(0,)),
    PSpec("e5_L120_drop_last", 5, 120, drop=(4,)),
    PSpec("e3_L2_tiny", 3, 2),
    PSpec("e2_L1_onesample", 2, 1),
]

#: plot() argument variations. ``n_epochs`` values that depend on the data are
#: spelled as sentinels resolved in :func:`resolve_kwargs`.
KWARGS = {
    "win2": dict(n_epochs=2),
    "win1": dict(n_epochs=1),
    "win_len": dict(n_epochs="LEN"),
    "win_big": dict(n_epochs="LEN_PLUS_5"),
    "events": dict(n_epochs=2, events=True),
    "butterfly": dict(n_epochs=2, butterfly=True),
    "decim1": dict(n_epochs=2, decim=1),
    "decim2": dict(n_epochs=2, decim=2),
    "decim4": dict(n_epochs=2, decim=4),
    "scalings_dict": dict(n_epochs=2, scalings=dict(eeg=20e-6)),
}

#: which kwargs to run for which specs (win* everywhere, the rest on a subset)
BROAD = ("win2", "win1", "win_len", "win_big")
DEEP_SPECS = (
    "e4_L100",
    "e4_L100_sf512p3_tneg",
    "e6_L100_drop_noncontig",
    "e4_L100_ch64_sf250_tneg",
    "e50_L40",
)
DEEP = ("events", "butterfly", "decim1", "decim2", "decim4", "scalings_dict")


def resolve_kwargs(kw, n_kept):
    out = dict(kw)
    if out.get("n_epochs") == "LEN":
        out["n_epochs"] = n_kept
    elif out.get("n_epochs") == "LEN_PLUS_5":
        out["n_epochs"] = n_kept + 5
    return out


def cases():
    """Yield ``(spec, kwargs_key)`` for the whole matrix."""
    for spec in SPECS:
        for key in BROAD:
            yield spec, key
        if spec.name in DEEP_SPECS:
            for key in DEEP:
                yield spec, key


# --------------------------------------------------------------------------
# construction (3-D fixed path only)
# --------------------------------------------------------------------------


def build_epochs(spec):
    from mne import EpochsArray, create_info

    rng = np.random.default_rng(0)
    names = [f"EEG{i:03d}" for i in range(spec.n_channels)]
    info = create_info(names, spec.sfreq, "eeg")
    data = (
        rng.standard_normal((spec.n_src, spec.n_channels, spec.n_samples)) * 1e-6
    )
    stride = spec.n_samples + int(spec.sfreq) + 100
    events = np.column_stack(
        [
            np.arange(spec.n_src) * stride + stride,
            np.zeros(spec.n_src, int),
            np.ones(spec.n_src, int),
        ]
    )
    epochs = EpochsArray(
        data,
        info,
        events=events,
        tmin=float(spec.tmin),
        event_id={"x": 1},
        baseline=None,
        verbose=False,
    )
    if spec.drop:
        epochs.drop(list(spec.drop), verbose=False)
    return epochs


def expected_boundaries(spec):
    """Boundaries computed from the spec, never from the figure."""
    lengths = np.full(spec.n_kept, spec.n_samples, int)
    bs = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    return bs, bs / spec.sfreq


# --------------------------------------------------------------------------
# packing state so a comparison can be exact
# --------------------------------------------------------------------------


def _fill(a):
    a = np.ma.filled(np.asarray(a), np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
    return np.ascontiguousarray(a, dtype=np.float64)


def hex_list(a):
    return [float(v).hex() for v in np.asarray(a, float).ravel()]


class Packer:
    """Collect big arrays into an npz sidecar, small ones inline as hex."""

    FULL_LIMIT = 20000

    def __init__(self):
        self.arrays = {}

    def pack(self, key, a):
        if a is None:
            return None
        a = _fill(a)
        digest = hashlib.sha256(a.tobytes()).hexdigest()
        rec = dict(shape=list(a.shape), sha=digest, n=int(a.size))
        flat = a.ravel()
        if a.size <= 64:
            rec["hex"] = hex_list(flat)
        else:
            rec["head"] = hex_list(flat[:8])
            rec["tail"] = hex_list(flat[-8:])
        if a.size <= self.FULL_LIMIT:
            self.arrays[key] = a
            rec["npz"] = key
        return rec


# --------------------------------------------------------------------------
# state capture (only attributes that exist in BOTH environments)
# --------------------------------------------------------------------------


def capture(fig, backend, packer, key):
    m = fig.mne
    st = {}

    def scalar(name, v):
        if v is None:
            st[name] = None
        elif isinstance(v, (bool, np.bool_)):
            st[name] = bool(v)
        elif isinstance(v, (int, np.integer)):
            st[name] = int(v)
        else:
            st[name] = float(v).hex()

    scalar("n_times", int(m.n_times))
    scalar("n_epochs", int(m.n_epochs))
    scalar("t_start", float(m.t_start))
    scalar("duration", float(m.duration))
    scalar("first_time", float(getattr(m, "first_time", 0.0)))
    scalar("n_channels", int(m.n_channels))
    scalar("ch_start", int(m.ch_start))
    scalar("butterfly", bool(m.butterfly))
    scalar("scale_factor", float(m.scale_factor))
    scalar("decim", int(m.decim) if np.isscalar(m.decim) else -1)
    scalar("sampling_period", float(m.sampling_period))
    st["bad_epochs"] = [int(v) for v in getattr(m, "bad_epochs", [])]
    st["bads"] = list(m.info["bads"])
    st["boundary_times"] = packer.pack(f"{key}|boundary_times", m.boundary_times)
    st["midpoints"] = packer.pack(f"{key}|midpoints", m.midpoints)

    # sample bounds and the data they select — the core of the claim
    try:
        start, stop = fig._get_start_stop()
        st["start"] = int(start)
        st["stop"] = int(stop)
        data, times = fig._load_data(start, stop)
        st["loaded_data"] = packer.pack(f"{key}|loaded_data", data)
        st["loaded_times"] = packer.pack(f"{key}|loaded_times", times)
    except Exception as exc:  # noqa: BLE001
        st["start_stop_error"] = f"{type(exc).__name__}: {exc}"

    st["data"] = packer.pack(f"{key}|data", m.data)
    st["times"] = packer.pack(f"{key}|times", m.times)

    # what the reader actually sees
    if backend == "matplotlib":
        patches = list(m.ax_hscroll.patches)
        st["hscroll_patches"] = packer.pack(
            f"{key}|hscroll_patches",
            np.array([[p.get_x(), p.get_width()] for p in patches], float)
            if patches
            else np.zeros((0, 2)),
        )
        segs = m.vline.get_segments()
        st["vline_segments"] = packer.pack(
            f"{key}|vline_segments",
            np.asarray(segs, float) if len(segs) else np.zeros((0, 2, 2)),
        )
        st["vline_text"] = str(m.vline_text.get_text())
        scalar("vline_visible", bool(m.vline_visible))
        st["xlim"] = hex_list(m.ax_main.get_xlim())
        st["ylim"] = hex_list(m.ax_main.get_ylim())
        tr_x, tr_y = [], []
        for tr in m.traces:
            tr_x.append(_fill(tr.get_xdata()))
            tr_y.append(_fill(tr.get_ydata()))
        st["trace_x"] = [
            packer.pack(f"{key}|trace_x{i}", a) for i, a in enumerate(tr_x)
        ]
        st["trace_y"] = [
            packer.pack(f"{key}|trace_y{i}", a) for i, a in enumerate(tr_y)
        ]
        st["n_epoch_traces"] = len(m.epoch_traces)
    else:
        st["view_range"] = hex_list(np.asarray(m.viewbox.viewRange(), float))
        st["epoch_idx"] = [int(v) for v in np.atleast_1d(m.epoch_idx)]
        scalar("epoch_dur", float(m.epoch_dur))
        scalar("xmax", float(m.xmax))
        if m.vline is None:
            st["vline_pos"] = None
            st["vline_vis"] = None
            st["vline_label"] = None
        else:
            st["vline_pos"] = hex_list([vl.value() for vl in m.vline])
            st["vline_vis"] = [bool(vl.isVisible()) for vl in m.vline]
            st["vline_label"] = [
                str(getattr(vl, "label", None).textItem.toPlainText())
                if getattr(vl, "label", None) is not None
                else None
                for vl in m.vline
            ]
        st["epoch_color_ref"] = packer.pack(
            f"{key}|epoch_color_ref", getattr(m, "epoch_color_ref", None)
        )
        sb = m.ax_hscroll
        st["hscroll"] = [
            int(sb.value()),
            int(sb.minimum()),
            int(sb.maximum()),
            int(sb.pageStep()),
        ]
        ob = getattr(m, "overview_bar", None)
        rects = getattr(ob, "bad_epoch_rect_dict", None) if ob is not None else None
        if rects is not None:
            st["ovb_bad_rects"] = [
                [
                    int(num),
                    *hex_list(
                        [
                            rects[num].rect().x(),
                            rects[num].rect().y(),
                            rects[num].rect().width(),
                            rects[num].rect().height(),
                        ]
                    ),
                ]
                for num in sorted(rects)
            ]
        if ob is not None and hasattr(ob, "_get_x_from_norm"):
            out = []
            for xn in (0.0, 0.125, 0.25, 0.5, 0.75, 0.999, 1.0):
                try:
                    out.append(repr(ob._get_x_from_norm(xn)))
                except Exception as exc:  # noqa: BLE001
                    out.append(f"ERR {type(exc).__name__}")
            st["ovb_x_from_norm"] = out
        tr_x, tr_y = [], []
        for tr in m.traces:
            tr_x.append(_fill(tr.xData if tr.xData is not None else []))
            tr_y.append(_fill(tr.yData if tr.yData is not None else []))
        st["trace_x"] = [
            packer.pack(f"{key}|trace_x{i}", a) for i, a in enumerate(tr_x)
        ]
        st["trace_y"] = [
            packer.pack(f"{key}|trace_y{i}", a) for i, a in enumerate(tr_y)
        ]
    return st


# --------------------------------------------------------------------------
# actions — identical gestures on both sides, positions from the spec
# --------------------------------------------------------------------------

MPL_SCRIPT = [
    "key:right",
    "key:right",
    "key:shift+right",
    "key:end",
    "key:end",
    "key:right",
    "key:home",
    "key:shift+left",
    "key:left",
    "click_bad:0",
    "key:right",
    "click_bad:1",
    "key:b",
    "click_vline:0.3",
    "click_vline:0.9",
    "key:pagedown",
    "key:pageup",
    "key:shift+right",
    "key:end",
    "key:home",
    "key:left",
    "key:left",
    "key:right",
]

QT_SCRIPT = [
    "hscroll:right",
    "hscroll:right",
    "hscroll:+full",
    "change_duration:+1",
    "change_duration:+1",
    "hscroll:right",
    "hscroll:-full",
    "hscroll:left",
    "vline:0.3",
    "qt_bad:0",
    "hscroll:right",
    "qt_bad:1",
    "key:b",
    "qt_bad:0",
    "change_duration:-1",
    "key:pagedown",
    "key:pageup",
    "key:end",
    "key:home",
    "vline:0.9",
    "hscroll:left",
    "key:right",
    "key:shift+right",
    "key:left",
]


def first_visible_ix(fig, bt):
    """Index of the first visible epoch, from t_start and the known boundaries."""
    return int(np.argmin(np.abs(np.asarray(bt[:-1], float) - float(fig.mne.t_start))))


def _epoch_x(bt, ix, frac):
    ix = int(np.clip(ix, 0, len(bt) - 2))
    return float(bt[ix] + frac * (bt[ix + 1] - bt[ix]))


def apply_action(fig, backend, name, bt):
    if name.startswith("key:"):
        fig._fake_keypress(name[4:])
        return
    if name.startswith("hscroll:"):
        fig.hscroll(name.split(":", 1)[1])
        return
    if name.startswith("change_duration:"):
        fig.change_duration(step=int(name.split(":", 1)[1]))
        return
    if name.startswith("vline:"):
        frac = float(name.split(":", 1)[1])
        ix = first_visible_ix(fig, bt)
        fig._add_vline(_epoch_x(bt, ix, frac))
        return
    if name.startswith("click_bad:"):
        which = int(name.split(":", 1)[1])
        ix = first_visible_ix(fig, bt) + which
        x = _epoch_x(bt, ix, 0.5)
        tr = fig.mne.traces[0]
        xd = _fill(tr.get_xdata())
        yd = _fill(tr.get_ydata())
        ok = np.isfinite(xd) & np.isfinite(yd)
        if not ok.any():
            return
        j = np.nanargmin(np.where(ok, np.abs(xd - x), np.inf))
        fig._fake_click((float(xd[j]), float(yd[j])), xform="data")
        return
    if name.startswith("qt_bad:"):
        # exactly what DataTrace.mouseClickEvent does on a left click
        which = int(name.split(":", 1)[1])
        ix = first_visible_ix(fig, bt) + which
        fig.mne.traces[0].toggle_bad(_epoch_x(bt, ix, 0.5))
        return
    if name.startswith("click_vline:"):
        frac = float(name.split(":", 1)[1])
        ix = first_visible_ix(fig, bt)
        x = _epoch_x(bt, ix, frac)
        y = float(fig.mne.trace_offsets[0]) + 0.45
        fig._fake_click((x, y), xform="data")
        return
    raise ValueError(f"unknown action {name!r}")


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------


def open_fig(epochs, backend, kwargs):
    import mne

    mne.viz.set_browser_backend(backend)
    kw = dict(show=False)
    kw.update(kwargs)
    if backend != "matplotlib":
        kw.update(precompute=False, use_opengl=False, theme="light")
    fig = epochs.plot(**kw)
    fig.test_mode = True
    return fig


def _flush(backend):
    if backend == "matplotlib":
        return
    from qtpy.QtWidgets import QApplication

    QApplication.processEvents()


def record_case(spec, kwargs_key, backend, packer, records):
    n_kept = spec.n_kept
    kwargs = resolve_kwargs(KWARGS[kwargs_key], n_kept)
    case_key = f"{spec.name}|{kwargs_key}|{backend}"
    _, bt = expected_boundaries(spec)
    script = MPL_SCRIPT if backend == "matplotlib" else QT_SCRIPT

    fig = None
    try:
        epochs = build_epochs(spec)
        fig = open_fig(epochs, backend, kwargs)
        _flush(backend)
    except Exception as exc:  # noqa: BLE001
        records.append(
            dict(
                case=case_key,
                step=0,
                action="<open>",
                open_error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=6),
            )
        )
        return
    try:
        records.append(
            dict(
                case=case_key,
                step=0,
                action="<open>",
                expected_boundary_times=hex_list(bt),
                state=capture(fig, backend, packer, f"{case_key}|0"),
            )
        )
        for step, name in enumerate(script, start=1):
            rec = dict(case=case_key, step=step, action=name)
            try:
                apply_action(fig, backend, name, bt)
                _flush(backend)
            except Exception as exc:  # noqa: BLE001
                rec["action_error"] = f"{type(exc).__name__}: {exc}"
            try:
                rec["state"] = capture(fig, backend, packer, f"{case_key}|{step}")
            except Exception as exc:  # noqa: BLE001
                rec["capture_error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
    finally:
        try:
            fig.close()
        except Exception:
            pass
        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            plt.close("all")


# --------------------------------------------------------------------------
# Raw / ICA side-check
# --------------------------------------------------------------------------

RAW_SCRIPT = [
    "key:right",
    "key:shift+right",
    "key:end",
    "key:end",
    "key:home",
    "key:left",
    "key:pageup",
    "key:pagedown",
    "key:b",
    "key:right",
    "key:right",
    "key:shift+left",
]


def build_raw(sfreq=512.3, n_ch=6, n_sec=12.0):
    from mne import create_info
    from mne.io import RawArray

    rng = np.random.default_rng(1)
    n = int(round(n_sec * sfreq))
    names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = create_info(names, sfreq, "eeg")
    return RawArray(rng.standard_normal((n_ch, n)) * 1e-6, info, verbose=False)


def build_epochs_from_raw(raw, n_ep=6, L=100, tmin=-0.2):
    from mne import Epochs

    sfreq = raw.info["sfreq"]
    stride = L + int(sfreq) // 2
    ev = np.column_stack(
        [
            np.arange(n_ep) * stride + int(sfreq),
            np.zeros(n_ep, int),
            np.ones(n_ep, int),
        ]
    )
    tmax = tmin + (L - 1) / sfreq
    return Epochs(
        raw,
        ev,
        {"x": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )


def build_ica(raw):
    from mne.preprocessing import ICA

    ica = ICA(n_components=4, max_iter=200, random_state=0, method="fastica")
    ica.fit(raw, verbose=False)
    return ica


def record_raw_ica(backend, packer, records):
    import mne

    mne.viz.set_browser_backend(backend)
    raw = build_raw()
    epochs = build_epochs_from_raw(raw)
    ica = build_ica(raw)

    def _run(label, opener, script, bt):
        fig = None
        key = f"{label}|{backend}"
        try:
            fig = opener()
            _flush(backend)
        except Exception as exc:  # noqa: BLE001
            records.append(
                dict(
                    case=key,
                    step=0,
                    action="<open>",
                    open_error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=6),
                )
            )
            return
        try:
            fig.test_mode = True
            records.append(
                dict(
                    case=key,
                    step=0,
                    action="<open>",
                    state=capture_generic(fig, backend, packer, f"{key}|0"),
                )
            )
            for step, name in enumerate(script, start=1):
                rec = dict(case=key, step=step, action=name)
                try:
                    apply_action(fig, backend, name, bt)
                    _flush(backend)
                except Exception as exc:  # noqa: BLE001
                    rec["action_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    rec["state"] = capture_generic(
                        fig, backend, packer, f"{key}|{step}"
                    )
                except Exception as exc:  # noqa: BLE001
                    rec["capture_error"] = f"{type(exc).__name__}: {exc}"
                records.append(rec)
        finally:
            try:
                fig.close()
            except Exception:
                pass
            if backend == "matplotlib":
                import matplotlib.pyplot as plt

                plt.close("all")

    sfreq = raw.info["sfreq"]
    ep_bt = np.arange(len(epochs) + 1) * 100 / sfreq

    _run(
        "raw",
        lambda: _open_generic(raw.plot, backend, dict(duration=3.0, n_channels=4)),
        RAW_SCRIPT,
        None,
    )
    _run(
        "ica_raw",
        lambda: _open_generic(
            lambda **kw: ica.plot_sources(raw, **kw), backend, dict()
        ),
        RAW_SCRIPT,
        None,
    )
    _run(
        "ica_epochs",
        lambda: _open_generic(
            lambda **kw: ica.plot_sources(epochs, **kw), backend, dict()
        ),
        MPL_SCRIPT if backend == "matplotlib" else QT_SCRIPT,
        ep_bt,
    )
    _run(
        "epochs_from_raw",
        lambda: _open_generic(epochs.plot, backend, dict(n_epochs=2)),
        MPL_SCRIPT if backend == "matplotlib" else QT_SCRIPT,
        ep_bt,
    )


def _open_generic(plot_fn, backend, kwargs):
    kw = dict(show=False)
    kw.update(kwargs)
    if backend != "matplotlib":
        kw.update(precompute=False, use_opengl=False, theme="light")
    return plot_fn(**kw)


def capture_generic(fig, backend, packer, key):
    """Capture for Raw/ICA: skip attributes that only exist for epochs."""
    m = fig.mne
    if m.is_epochs:
        return capture(fig, backend, packer, key)
    st = {}
    st["n_times"] = int(m.n_times)
    st["t_start"] = float(m.t_start).hex()
    st["duration"] = float(m.duration).hex()
    st["first_time"] = float(m.first_time).hex()
    st["n_channels"] = int(m.n_channels)
    st["ch_start"] = int(m.ch_start)
    st["butterfly"] = bool(m.butterfly)
    st["sampling_period"] = float(m.sampling_period).hex()
    st["bads"] = list(m.info["bads"])
    try:
        start, stop = fig._get_start_stop()
        st["start"], st["stop"] = int(start), int(stop)
        data, times = fig._load_data(start, stop)
        st["loaded_data"] = packer.pack(f"{key}|loaded_data", data)
        st["loaded_times"] = packer.pack(f"{key}|loaded_times", times)
    except Exception as exc:  # noqa: BLE001
        st["start_stop_error"] = f"{type(exc).__name__}: {exc}"
    st["data"] = packer.pack(f"{key}|data", m.data)
    st["times"] = packer.pack(f"{key}|times", m.times)
    if backend == "matplotlib":
        st["xlim"] = hex_list(m.ax_main.get_xlim())
        st["trace_x"] = [
            packer.pack(f"{key}|trace_x{i}", _fill(tr.get_xdata()))
            for i, tr in enumerate(m.traces)
        ]
        st["trace_y"] = [
            packer.pack(f"{key}|trace_y{i}", _fill(tr.get_ydata()))
            for i, tr in enumerate(m.traces)
        ]
    else:
        st["view_range"] = hex_list(np.asarray(m.viewbox.viewRange(), float))
        st["xmax"] = float(m.xmax).hex()
        ob = getattr(m, "overview_bar", None)
        if ob is not None and hasattr(ob, "_get_x_from_norm"):
            st["ovb_x_from_norm"] = [
                repr(ob._get_x_from_norm(xn))
                for xn in (0.0, 0.25, 0.5, 0.75, 0.999, 1.0)
            ]
        st["trace_x"] = [
            packer.pack(f"{key}|trace_x{i}", _fill(tr.xData))
            for i, tr in enumerate(m.traces)
        ]
        st["trace_y"] = [
            packer.pack(f"{key}|trace_y{i}", _fill(tr.yData))
            for i, tr in enumerate(m.traces)
        ]
    return st


# --------------------------------------------------------------------------
# precompute sub-matrix (qt only)
# --------------------------------------------------------------------------

PRECOMPUTE_SPECS = ("e4_L100", "e4_L100_sf512p3_tneg", "e6_L100_drop_noncontig")


def record_precompute(backend, packer, records):
    if backend == "matplotlib":
        return
    from qtpy.QtTest import QTest

    for spec in SPECS:
        if spec.name not in PRECOMPUTE_SPECS:
            continue
        case_key = f"precompute|{spec.name}|{backend}"
        _, bt = expected_boundaries(spec)
        fig = None
        try:
            import mne

            mne.viz.set_browser_backend(backend)
            epochs = build_epochs(spec)
            fig = epochs.plot(
                n_epochs=2,
                show=False,
                precompute=True,
                use_opengl=False,
                theme="light",
            )
            fig.test_mode = True
            for _ in range(300):
                if fig.mne.data_precomputed:
                    break
                QTest.qWait(50)
            records.append(
                dict(
                    case=case_key,
                    step=0,
                    action="<open>",
                    precomputed=bool(fig.mne.data_precomputed),
                    global_times=packer.pack(
                        f"{case_key}|global_times",
                        getattr(fig.mne, "global_times", None),
                    ),
                    state=capture(fig, backend, packer, f"{case_key}|0"),
                )
            )
            for step, name in enumerate(QT_SCRIPT[:8], start=1):
                rec = dict(case=case_key, step=step, action=name)
                try:
                    apply_action(fig, backend, name, bt)
                    _flush(backend)
                except Exception as exc:  # noqa: BLE001
                    rec["action_error"] = f"{type(exc).__name__}: {exc}"
                rec["state"] = capture(fig, backend, packer, f"{case_key}|{step}")
                records.append(rec)
        except Exception as exc:  # noqa: BLE001
            records.append(
                dict(
                    case=case_key,
                    step=-1,
                    action="<fatal>",
                    open_error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=6),
                )
            )
        finally:
            try:
                fig.close()
            except Exception:
                pass




# --------------------------------------------------------------------------
# broader Raw / ICA sub-matrix (A7 second pass)
# --------------------------------------------------------------------------

#: (name, sfreq, n_ch, n_sec, first_samp)
RAW_SPECS = [
    ("raw_sf100", 100.0, 6, 12.0, 0),
    ("raw_sf250_fs", 250.0, 6, 12.0, 733),
    ("raw_sf512p3", 512.3, 6, 12.0, 0),
    ("raw_sf512p3_fs", 512.3, 6, 12.0, 1001),
    ("raw_sf1000_ch1", 1000.0, 1, 6.0, 0),
    ("raw_sf250_ch64", 250.0, 64, 8.0, 0),
]

#: plot() kwargs for Raw
RAW_KWARGS = {
    "d3": dict(duration=3.0, n_channels=4),
    "d_all": dict(duration=1e6, n_channels=4),
    "d3_bf": dict(duration=3.0, n_channels=4, butterfly=True),
    "d3_dec2": dict(duration=3.0, n_channels=4, decim=2),
    "d3_dec4": dict(duration=3.0, n_channels=4, decim=4),
    "d3_sc": dict(duration=3.0, n_channels=4, scalings=dict(eeg=20e-6)),
}

#: equal-duration epochs cut from a Raw, for the ICA-on-epochs path
ICA_EPOCH_SPECS = [
    ("ie_sf100_L100_t0", 100.0, 6, 100, 0.0),
    ("ie_sf250_L120_tneg", 250.0, 6, 120, -0.2),
    ("ie_sf512p3_L100_tneg", 512.3, 6, 100, -0.2),
    ("ie_sf1000_L200_tpos", 1000.0, 6, 200, 0.1),
]

RAW_SCRIPT_QT = [
    "key:right",
    "hscroll:right",
    "hscroll:+full",
    "key:end",
    "key:home",
    "key:left",
    "change_duration:+1",
    "hscroll:right",
    "key:b",
    "change_duration:-1",
    "key:pagedown",
    "key:pageup",
]


def build_raw2(sfreq, n_ch, n_sec, first_samp):
    from mne import create_info
    from mne.io import RawArray

    rng = np.random.default_rng(1)
    n = int(round(n_sec * sfreq))
    info = create_info([f"EEG{i:03d}" for i in range(n_ch)], sfreq, "eeg")
    return RawArray(
        rng.standard_normal((n_ch, n)) * 1e-6,
        info,
        first_samp=first_samp,
        verbose=False,
    )


def _close(fig, backend):
    try:
        fig.close()
    except Exception:
        pass
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close("all")


def _drive(fig, backend, script, bt, key, packer, records):
    """Capture the open state then replay ``script``, capturing each step."""
    fig.test_mode = True
    records.append(
        dict(
            case=key,
            step=0,
            action="<open>",
            state=capture_generic(fig, backend, packer, f"{key}|0"),
        )
    )
    for step, name in enumerate(script, start=1):
        rec = dict(case=key, step=step, action=name)
        try:
            apply_action(fig, backend, name, bt)
            _flush(backend)
        except Exception as exc:  # noqa: BLE001
            rec["action_error"] = f"{type(exc).__name__}: {exc}"
        try:
            rec["state"] = capture_generic(fig, backend, packer, f"{key}|{step}")
        except Exception as exc:  # noqa: BLE001
            rec["capture_error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)


def record_raw_matrix(backend, packer, records):
    """Raw browsing across sfreq / first_samp / channel count / plot kwargs."""
    import mne

    mne.viz.set_browser_backend(backend)
    script = RAW_SCRIPT if backend == "matplotlib" else RAW_SCRIPT_QT
    for name, sfreq, n_ch, n_sec, first_samp in RAW_SPECS:
        raw = build_raw2(sfreq, n_ch, n_sec, first_samp)
        for kw_key, kw in RAW_KWARGS.items():
            key = f"rawm|{name}|{kw_key}|{backend}"
            print(f"  {key}", flush=True)
            fig = None
            try:
                fig = _open_generic(raw.plot, backend, kw)
                _flush(backend)
            except Exception as exc:  # noqa: BLE001
                records.append(
                    dict(
                        case=key,
                        step=0,
                        action="<open>",
                        open_error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(limit=6),
                    )
                )
                continue
            try:
                _drive(fig, backend, script, None, key, packer, records)
            finally:
                _close(fig, backend)


def record_ica_matrix(backend, packer, records):
    """ICA source browsing on Raw and on equal-duration Epochs."""
    import mne
    from mne.preprocessing import ICA

    mne.viz.set_browser_backend(backend)
    ep_script = MPL_SCRIPT if backend == "matplotlib" else QT_SCRIPT
    raw_script = RAW_SCRIPT if backend == "matplotlib" else RAW_SCRIPT_QT

    for name, sfreq, n_ch, L, tmin in ICA_EPOCH_SPECS:
        raw = build_raw2(sfreq, n_ch, 14.0, 0)
        ica = ICA(n_components=4, max_iter=200, random_state=0, method="fastica")
        ica.fit(raw, verbose=False)
        epochs = build_epochs_from_raw(raw, n_ep=6, L=L, tmin=tmin)
        bt = np.arange(len(epochs) + 1) * L / sfreq

        key = f"icam|{name}|raw|{backend}"
        print(f"  {key}", flush=True)
        fig = None
        try:
            fig = _open_generic(lambda **kw: ica.plot_sources(raw, **kw), backend, {})
            _flush(backend)
            _drive(fig, backend, raw_script, None, key, packer, records)
        except Exception as exc:  # noqa: BLE001
            records.append(
                dict(
                    case=key,
                    step=0,
                    action="<open>",
                    open_error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=6),
                )
            )
        finally:
            if fig is not None:
                _close(fig, backend)

        key = f"icam|{name}|epochs|{backend}"
        print(f"  {key}", flush=True)
        fig = None
        try:
            fig = _open_generic(
                lambda **kw: ica.plot_sources(epochs, **kw), backend, {}
            )
            _flush(backend)
            _drive(fig, backend, ep_script, bt, key, packer, records)
        except Exception as exc:  # noqa: BLE001
            records.append(
                dict(
                    case=key,
                    step=0,
                    action="<open>",
                    open_error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=6),
                )
            )
        finally:
            if fig is not None:
                _close(fig, backend)

        # ICA fitted on the equal-duration epochs themselves, not on the Raw
        key = f"icam|{name}|fitepochs|{backend}"
        print(f"  {key}", flush=True)
        fig = None
        try:
            ica2 = ICA(n_components=4, max_iter=200, random_state=0, method="fastica")
            ica2.fit(epochs, verbose=False)
            fig = _open_generic(
                lambda **kw: ica2.plot_sources(epochs, **kw), backend, {}
            )
            _flush(backend)
            _drive(fig, backend, ep_script, bt, key, packer, records)
        except Exception as exc:  # noqa: BLE001
            records.append(
                dict(
                    case=key,
                    step=0,
                    action="<open>",
                    open_error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=6),
                )
            )
        finally:
            if fig is not None:
                _close(fig, backend)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def cmd_record(args):
    os.makedirs(args.out, exist_ok=True)
    env = env_report()
    packer = Packer()
    records = []
    sec = args.section
    if sec in ("all", "epochs"):
        todo = list(cases())
        if args.only:
            todo = [c for c in todo if c[0].name == args.only]
        for i, (spec, key) in enumerate(todo):
            print(f"[{i + 1}/{len(todo)}] {spec.name} {key} {args.backend}", flush=True)
            record_case(spec, key, args.backend, packer, records)
    if sec == "all" and not args.only:
        print("raw / ica", flush=True)
        record_raw_ica(args.backend, packer, records)
        print("precompute", flush=True)
        record_precompute(args.backend, packer, records)
    if sec in ("all", "rawica"):
        print("raw matrix", flush=True)
        record_raw_matrix(args.backend, packer, records)
        print("ica matrix", flush=True)
        record_ica_matrix(args.backend, packer, records)
    tag = "" if args.section == "all" else f"_{args.section}"
    stem = os.path.join(args.out, f"{env['flavour']}{tag}_{args.backend}")
    with open(stem + ".json", "w") as fid:
        json.dump(dict(env=env, records=records), fid)
    np.savez_compressed(stem + ".npz", **packer.arrays)
    print(f"wrote {stem}.json ({len(records)} records) and {stem}.npz")


def _load(path, backend):
    want = {f"base_{backend}.json", f"branch_{backend}.json"}
    stems = [os.path.join(path, f) for f in os.listdir(path) if f in want]
    if len(stems) != 1:
        raise SystemExit(f"expected one of {sorted(want)} in {path}, got {stems}")
    with open(stems[0]) as fid:
        blob = json.load(fid)
    arrays = np.load(stems[0][:-5] + ".npz")
    return blob, arrays


def _cmp_array(a_rec, b_rec, a_np, b_np, label, out):
    if a_rec is None and b_rec is None:
        return
    if (a_rec is None) != (b_rec is None):
        out.append(f"{label}: present on one side only ({a_rec} vs {b_rec})")
        return
    if a_rec["shape"] != b_rec["shape"]:
        out.append(f"{label}: shape {a_rec['shape']} vs {b_rec['shape']}")
        return
    if a_rec["sha"] == b_rec["sha"]:
        return
    if "npz" in a_rec and "npz" in b_rec and a_rec["npz"] in a_np.files:
        a = a_np[a_rec["npz"]]
        b = b_np[b_rec["npz"]]
        eq = np.array_equal(a, b) or (
            np.array_equal(np.isnan(a), np.isnan(b))
            and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        )
        if eq:
            return  # only a -0.0 / NaN-payload difference
        d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
        i = int(np.argmax(d))
        out.append(
            f"{label}: differs; max|diff|={d.max():.6g} at flat index {i} "
            f"(base {a.ravel()[i]!r} vs branch {b.ravel()[i]!r}); "
            f"n_differing={int((d > 0).sum())}/{a.size}"
        )
    else:
        out.append(
            f"{label}: sha differs ({a_rec['sha'][:12]} vs {b_rec['sha'][:12]}), "
            f"head {a_rec.get('head')} vs {b_rec.get('head')}"
        )


ARRAY_FIELDS = (
    "epoch_color_ref",
    "boundary_times",
    "midpoints",
    "loaded_data",
    "loaded_times",
    "data",
    "times",
    "hscroll_patches",
    "vline_segments",
)


def _cmp_state(sa, sb, a_np, b_np, out):
    keys = sorted(set(sa) | set(sb))
    for k in keys:
        if k not in sa or k not in sb:
            out.append(f"{k}: only in {'base' if k in sa else 'branch'}")
            continue
        va, vb = sa[k], sb[k]
        if k in ARRAY_FIELDS:
            _cmp_array(va, vb, a_np, b_np, k, out)
        elif k in ("trace_x", "trace_y"):
            if len(va) != len(vb):
                out.append(f"{k}: {len(va)} traces vs {len(vb)}")
                continue
            for i, (ra, rb) in enumerate(zip(va, vb)):
                _cmp_array(ra, rb, a_np, b_np, f"{k}[{i}]", out)
        elif va != vb:
            out.append(f"{k}: base={va!r} branch={vb!r}")


def cmd_diff(args):
    for backend in args.backends:
        a_blob, a_np = _load(args.base, backend)
        b_blob, b_np = _load(args.branch, backend)
        print(f"\n===== {backend} =====")
        print(f"base   : {a_blob['env']['flavour']} {a_blob['env']['mne_file']}")
        print(f"branch : {b_blob['env']['flavour']} {b_blob['env']['mne_file']}")
        a_recs = {(r["case"], r["step"]): r for r in a_blob["records"]}
        b_recs = {(r["case"], r["step"]): r for r in b_blob["records"]}
        cases_seen = sorted({c for c, _ in a_recs} | {c for c, _ in b_recs})
        n_states = 0
        n_bad = 0
        buckets = {}
        for case in cases_seen:
            steps = sorted(
                {s for c, s in a_recs if c == case} | {s for c, s in b_recs if c == case}
            )
            reported = 0
            for step in steps:
                ra = a_recs.get((case, step))
                rb = b_recs.get((case, step))
                n_states += 1
                out = []
                if ra is None or rb is None:
                    out.append(f"record missing on {'base' if ra is None else 'branch'}")
                else:
                    for k in ("open_error", "action_error", "capture_error"):
                        if ra.get(k) != rb.get(k):
                            out.append(f"{k}: base={ra.get(k)!r} branch={rb.get(k)!r}")
                    if "state" in ra and "state" in rb:
                        _cmp_state(ra["state"], rb["state"], a_np, b_np, out)
                    elif ("state" in ra) != ("state" in rb):
                        out.append("state captured on one side only")
                if out:
                    n_bad += 1
                    for line in out:
                        field = line.split(":", 1)[0].split("[")[0]
                        buckets.setdefault(field, []).append(
                            (case, step, ra["action"] if ra else "", line)
                        )
                    if args.summary:
                        continue
                    if reported < args.max_per_case:
                        print(f"\n-- {case}  step {step}  {ra['action'] if ra else ''}")
                        for line in out[: args.max_lines]:
                            print(f"     {line}")
                        if len(out) > args.max_lines:
                            print(f"     ... {len(out) - args.max_lines} more")
                        reported += 1
                    elif reported == args.max_per_case:
                        print(f"-- {case}: further divergences suppressed")
                        reported += 1
        if buckets:
            print(f"\n--- {backend}: divergences by field ---")
            for field, rows in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
                ncase = len({r[0] for r in rows})
                print(f"\n[{field}]  {len(rows)} states, {ncase} cases")
                for c, s, a, line in rows[: args.examples]:
                    print(f"    {c} step {s} ({a})")
                    print(f"      {line}")
                if len(rows) > args.examples:
                    print(f"    ... {len(rows) - args.examples} more")
                    cs = sorted({r[0] for r in rows})
                    print(f"    cases: {', '.join(cs[:12])}"
                          + (f" (+{len(cs) - 12})" if len(cs) > 12 else ""))
        print(
            f"\n{backend}: {n_states} states compared, {n_bad} with divergences, "
            f"{len(cases_seen)} cases"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--backend", required=True, choices=["matplotlib", "qt"])
    r.add_argument("--out", required=True)
    r.add_argument("--only", default=None)
    r.add_argument("--section", default="all", choices=["all", "epochs", "rawica"])
    r.set_defaults(func=cmd_record)

    d = sub.add_parser("diff")
    d.add_argument("base")
    d.add_argument("branch")
    d.add_argument("--backends", nargs="+", default=["matplotlib", "qt"])
    d.add_argument("--max-per-case", type=int, default=3)
    d.add_argument("--max-lines", type=int, default=12)
    d.add_argument("--summary", action="store_true")
    d.add_argument("--examples", type=int, default=3)
    d.set_defaults(func=cmd_diff)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
