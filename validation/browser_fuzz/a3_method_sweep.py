"""A3-6 — what every lifecycle-adjacent public method does on ragged epochs.

Sorts each call into: native (ran, bounds intact), declined (a clear
NotImplementedError from the PR's own tables), fallback (RuntimeWarning then
as_fixed), or LEAK -- an internal exception the PR never chose to raise.
"""

import traceback
import warnings

import numpy as np

import mne
from a3_lifecycle import audit, banner, make

LENGTHS = (100, 250, 75, 180, 120)
TMINS = (0.0, -0.2, 0.1, -0.5, 0.3)


def fresh(**kw):
    return make(LENGTHS, TMINS, **kw)


def fresh_fixed():
    from mne import EpochsArray, create_info

    info = create_info(["EEG000", "EEG001", "EEG002"], 100.0, ["eeg"] * 3)
    rng = np.random.default_rng(0)
    d = rng.standard_normal((5, 3, 100)) * 1e-6
    ev = np.column_stack(
        [np.arange(5) * 300 + 300, np.zeros(5, int), np.ones(5, int)]
    )
    return EpochsArray(
        d, info, events=ev, tmin=0.0, event_id={"e1": 1}, baseline=None, verbose=False
    )


CALLS = [
    ("copy", lambda e: e.copy()),
    ("__getitem__[1:3]", lambda e: e[1:3]),
    ("drop([1])", lambda e: e.drop([1], verbose=False)),
    ("drop_bad()", lambda e: e.drop_bad(verbose=False)),
    ("drop_bad(reject=)", lambda e: e.drop_bad(reject=dict(eeg=200e-6), verbose=False)),
    ("drop_bad(flat=)", lambda e: e.drop_bad(flat=dict(eeg=1e-15), verbose=False)),
    ("equalize_event_counts", lambda e: e.equalize_event_counts(["e1"])),
    ("crop(0, 0.5)", lambda e: e.crop(0.0, 0.5)),
    ("shift_time(0.3)", lambda e: e.shift_time(0.3)),
    ("pick(['EEG000'])", lambda e: e.pick(["EEG000"])),
    ("drop_channels(['EEG000'])", lambda e: e.drop_channels(["EEG000"])),
    ("reorder_channels", lambda e: e.reorder_channels(["EEG002", "EEG001", "EEG000"])),
    ("rename_channels", lambda e: e.rename_channels({"EEG000": "Z"})),
    ("set_channel_types", lambda e: e.set_channel_types({"EEG000": "misc"},
                                                        verbose=False)),
    ("get_data()", lambda e: e.get_data(copy=False)),
    ("get_data(item=[0,1])", lambda e: e.get_data(item=[0, 1], copy=False)),
    ("iter (next)", lambda e: next(iter(e))),
    ("as_fixed()", lambda e: e.as_fixed()),
    ("to_data_frame()", lambda e: e.to_data_frame()),
    ("average()", lambda e: e.average()),
    ("filter(1, 40)", lambda e: e.filter(1, 40, verbose=False)),
    ("resample(50)", lambda e: e.resample(50, verbose=False)),
    ("decimate(2)", lambda e: e.decimate(2, verbose=False)),
    ("apply_baseline", lambda e: e.apply_baseline((None, 0), verbose=False)),
    ("set_eeg_reference", lambda e: e.set_eeg_reference("average", verbose=False)),
    ("apply_function", lambda e: e.apply_function(lambda x: x * 2)),
    ("interpolate_bads", lambda e: e.interpolate_bads(verbose=False)),
    ("add_channels", lambda e: e.add_channels([e.copy().pick(["EEG000"]).rename_channels(
        {"EEG000": "NEW"})])),
    ("time_as_index(0.5)", lambda e: e.time_as_index(0.5)),
    ("savgol_filter(20)", lambda e: e.savgol_filter(20, verbose=False)),
    ("standard_error()", lambda e: e.standard_error()),
    ("subtract_evoked()", lambda e: e.subtract_evoked()),
    ("compute_psd()", lambda e: e.compute_psd(verbose=False)),
    ("plot_drop_log()", lambda e: e.plot_drop_log(show=False)),
    ("plot_image()", lambda e: e.plot_image(show=False)),
    ("plot_psd()", lambda e: e.plot_psd(show=False)),
    ("concatenate_epochs([e,e])",
     lambda e: mne.concatenate_epochs([e.copy(), e.copy()])),
    ("mne.epochs.combine_event_ids",
     lambda e: mne.epochs.combine_event_ids(e, ["e1"], {"m": 99})),
]


def classify(call, ep):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = call(ep)
        except NotImplementedError as exc:
            return "declined", f"NotImplementedError: {str(exc)[:80]}", None
        except Exception as exc:
            tb = traceback.format_exc().splitlines()
            frame = [ln.strip() for ln in tb if ln.strip().startswith("File ")]
            where = frame[-1] if frame else ""
            return "LEAK", f"{type(exc).__name__}: {str(exc)[:100]}", where
        msgs = [str(w.message) for w in caught]
        fell_back = any("ran on as_fixed()" in m for m in msgs)
        return ("fallback" if fell_back else "native"), "", out


def main():
    banner("method sweep on ragged epochs")
    rows = []
    for name, call in CALLS:
        ep = fresh()
        verdict, detail, out = classify(call, ep)
        extra = ""
        if verdict == "native":
            target = out if isinstance(out, mne.BaseEpochs) else ep
            if isinstance(target, mne.BaseEpochs) and getattr(
                target, "_variable_duration", False
            ):
                bad = audit(target, name)
                if bad:
                    verdict = "NATIVE-BUT-BROKEN"
                    extra = "; ".join(bad)[:160]
        rows.append((name, verdict, detail or extra))

    banner("fixed-path control for anything that leaked")
    fixed_note = {}
    for name, call in CALLS:
        entry = [r for r in rows if r[0] == name][0]
        if entry[1] != "LEAK":
            continue
        ep = fresh_fixed()
        v, d, _ = classify(call, ep)
        fixed_note[name] = f"{v} {d}"

    banner("RESULT")
    width = max(len(n) for n, _, _ in rows)
    for name, verdict, detail in rows:
        mark = "  " if verdict in ("native", "declined", "fallback") else ">>"
        print(f"{mark} {name:<{width}}  {verdict:<18} {detail}")
        if name in fixed_note:
            print(f"{'':<{width + 25}}fixed path: {fixed_note[name]}")

    leaks = [r for r in rows if r[1] in ("LEAK", "NATIVE-BUT-BROKEN")]
    banner(f"{len(leaks)} methods leak an internal error or corrupt the bounds")
    for name, verdict, detail in leaks:
        print(f"  {name}: {verdict} {detail}")
        if name in fixed_note:
            print(f"      fixed path -> {fixed_note[name]}")


if __name__ == "__main__":
    main()
