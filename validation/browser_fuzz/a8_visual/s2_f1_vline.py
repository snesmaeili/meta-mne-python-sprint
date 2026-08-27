"""A8-2: what F1 looks like on screen, before and after a scroll (Qt).

Ragged (100,250,75,180) @100 Hz, boundaries [0, 1, 3.5, 4.25, 6.05], n_epochs=2.
A vline is placed at latency 0.900 s in epoch 0, then the view scrolls one epoch
right so that epoch 2 -- only 0.75 s long -- becomes visible. Epoch 2 never
reaches 0.900 s, so its line should be hidden.

Also renders the equal-duration control through the same gesture.
"""

import numpy as np

from vis import app, build, open_qt, qt_numbers, rep_spec, shot_qt


def vline_report(fig, case, tag):
    sfreq = case.sfreq
    print(f"  vlines {tag}:")
    if fig.mne.vline is None:
        print("    (none)")
        return []
    ix0, ix1 = fig._get_epoch_ix_range()
    rows = []
    for slot, vl in enumerate(fig.mne.vline):
        x = float(vl.value())
        idx = int(
            np.clip(
                np.searchsorted(case.boundary_times[1:], x, side="right"),
                0,
                case.n_epochs - 1,
            )
        )
        lat = case.tmins[idx] + round((x - case.boundary_times[idx]) * sfreq) / sfreq
        dur = case.lengths[idx] / sfreq
        reaches = lat <= case.tmins[idx] + (case.lengths[idx] - 1) / sfreq + 1e-9
        try:
            shown = vl.label.textItem.toPlainText()
        except Exception:
            shown = "?"
        print(
            f"    slot {slot}: x={x:.4f}s  epoch {idx} (dur {dur:.2f}s)  "
            f"latency={lat:.4f}s  visible={vl.isVisible()}  label={shown!r}  "
            f"epoch_reaches_latency={reaches}"
        )
        rows.append((slot, x, idx, lat, vl.isVisible(), reaches))
    print(f"    epoch_ix_range={ (ix0, ix1) }  epoch_idx={np.asarray(fig.mne.epoch_idx).tolist()}")
    return rows


for label, spec in [
    ("ragged", rep_spec((100, 250, 75, 180), n_channels=20)),
    ("fixed", rep_spec((150, 150, 150, 150), n_channels=20, force_fixed=True)),
]:
    print(f"\n=== F1 {label} ===")
    case = build(spec)
    fig = open_qt(case, n_epochs=2)
    qt_numbers(fig, case, "at open")

    fig._add_vline(0.9)
    app().processEvents()
    shot_qt(fig, f"a8_f1_{label}_1_before_scroll.png")
    vline_report(fig, case, "after _add_vline(0.9)")
    print(f"    view={[round(float(v), 4) for v in fig.mne.viewbox.viewRange()[0]]}")

    fig.hscroll("right")
    app().processEvents()
    shot_qt(fig, f"a8_f1_{label}_2_after_scroll.png")
    vline_report(fig, case, "after hscroll('right')")
    print(f"    view={[round(float(v), 4) for v in fig.mne.viewbox.viewRange()[0]]}")

    # A4-3(a): the corruption persists after scrolling back
    fig.hscroll("left")
    app().processEvents()
    shot_qt(fig, f"a8_f1_{label}_3_scrolled_back.png")
    vline_report(fig, case, "after hscroll('left') back to the first view")
    print(f"    view={[round(float(v), 4) for v in fig.mne.viewbox.viewRange()[0]]}")
    fig.close()
