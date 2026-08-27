"""A3-3 — shift_time, pick/drop_channels, copy, equalize, concatenate, drop_bad."""

import numpy as np

import mne
from a3_lifecycle import TAG_UNIT, audit, banner, lengths_of, make, tags

SFREQ = 100.0
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL", msg)


def ok(msg):
    print("  ok  ", msg)


def show(ep, label):
    print(
        f"       {label}: tmin={np.round(ep._tmin_per_epoch, 4)} "
        f"tmax={np.round(ep._tmax_per_epoch, 4)} lens={lengths_of(ep)} "
        f"raw_times=[{ep._raw_times[0]:.4f}, {ep._raw_times[-1]:.4f}] "
        f"n={len(ep._raw_times)}"
    )


# --------------------------------------------------------------- shift_time
def test_shift_time():
    banner("shift_time")
    lengths = (100, 250, 75, 180)
    for tmins, name in [
        ((0.0, 0.0, 0.0, 0.0), "tmin all zero"),
        ((0.0, -0.2, 0.1, -0.5), "tmin mixed sign"),
        ((-0.5, -0.4, -0.3, -0.2), "tmin all negative"),
        ((0.2, 0.3, 0.4, 0.5), "tmin all positive"),
    ]:
        for relative, tshift in [(True, 0.3), (True, -0.3), (False, 0.0), (False, 1.0)]:
            ep = make(lengths, tmins)
            t0 = np.array(ep._tmin_per_epoch, copy=True)
            t1 = np.array(ep._tmax_per_epoch, copy=True)
            ep.shift_time(tshift, relative=relative)
            if relative:
                want0, want1 = t0 + tshift, t1 + tshift
            else:
                # "shift the time values such that the time of the first sample
                # equals tshift" -- for ragged epochs each epoch has its own
                # first sample, so the only self-consistent reading is that the
                # earliest one lands on tshift and the offsets are preserved
                want0, want1 = t0 + (tshift - t0.min()), t1 + (tshift - t0.min())
            lbl = f"{name}, relative={relative}, tshift={tshift}"
            msgs = audit(ep, lbl, expect_tags=[0, 1, 2, 3])
            if not np.allclose(ep._tmin_per_epoch, want0, atol=1e-12):
                msgs.append(f"{lbl}: tmin {ep._tmin_per_epoch} != {want0}")
            if not np.allclose(ep._tmax_per_epoch, want1, atol=1e-12):
                msgs.append(f"{lbl}: tmax {ep._tmax_per_epoch} != {want1}")
            # _raw_times must still be the union window on the sample grid
            wr0 = round(want0.min() * SFREQ)
            wr1 = round(want1.max() * SFREQ)
            if not np.isclose(ep._raw_times[0], wr0 / SFREQ, atol=1e-9) or not np.isclose(
                ep._raw_times[-1], wr1 / SFREQ, atol=1e-9
            ):
                msgs.append(
                    f"{lbl}: _raw_times [{ep._raw_times[0]:.4f},"
                    f"{ep._raw_times[-1]:.4f}] != union "
                    f"[{wr0 / SFREQ:.4f},{wr1 / SFREQ:.4f}]"
                )
            # get_times must agree with the bounds and the sample count
            for i in range(4):
                gt = ep.get_times(i)
                if len(gt) != lengths[i]:
                    msgs.append(
                        f"{lbl}: get_times({i}) has {len(gt)} entries, "
                        f"epoch has {lengths[i]} samples"
                    )
                elif not np.isclose(gt[0], ep._tmin_per_epoch[i], atol=1e-9):
                    msgs.append(f"{lbl}: get_times({i})[0]={gt[0]} != tmin")
            for m in msgs:
                fail(m)
            if not msgs:
                ok(f"{lbl}: tmin -> {np.round(ep._tmin_per_epoch, 4)}")

    banner("shift_time then plot: do the browser boundaries still land right?")
    for backend in ("matplotlib", "qt"):
        ep = make(lengths, (0.0, -0.2, 0.1, -0.5))
        ep.shift_time(0.75, relative=True)
        mne.viz.set_browser_backend(backend)
        fig = ep.plot(n_epochs=2, show=False)
        fig.test_mode = True
        bs = np.concatenate([[0], np.cumsum(lengths)])
        bt = bs / SFREQ
        msgs = []
        if not np.allclose(fig.mne.boundary_times, bt):
            msgs.append(f"{backend}: boundary_times {fig.mne.boundary_times} != {bt}")
        if not np.array_equal(np.asarray(fig.mne.boundary_samples), bs):
            msgs.append(f"{backend}: boundary_samples != {bs}")
        # epoch_tmins is what the vline latency arithmetic reads
        et = np.asarray(getattr(fig.mne, "epoch_tmins", []))
        if len(et) and not np.allclose(et, ep._tmin_per_epoch):
            msgs.append(f"{backend}: epoch_tmins {et} != {ep._tmin_per_epoch}")
        for m in msgs:
            fail(m)
        if not msgs:
            ok(f"{backend}: boundaries and epoch_tmins correct after shift_time")
        try:
            fig.close()
        except Exception:
            pass


# ------------------------------------------------------- pick / drop_channels
def test_channels():
    banner("pick / drop_channels aliasing")
    lengths = (100, 250, 75, 180)
    for op_name, op in [
        ("pick(['EEG001','EEG002'])", lambda e: e.pick(["EEG001", "EEG002"])),
        ("drop_channels(['EEG001'])", lambda e: e.drop_channels(["EEG001"])),
        ("pick('eeg')", lambda e: e.pick("eeg")),
        ("reorder_channels reversed", lambda e: e.reorder_channels(
            ["EEG003", "EEG002", "EEG001", "EEG000"])),
    ]:
        ep = make(lengths, n_channels=4)
        snapshot = ep.copy()
        pre_lists = [d.copy() for d in ep._data]
        pre_id = id(ep._data)
        try:
            op(ep)
        except Exception as exc:
            fail(f"{op_name}: raised {type(exc).__name__}: {exc}")
            continue
        msgs = audit(ep, op_name)
        # the copy taken beforehand must be untouched
        if len(snapshot._data) != len(lengths):
            msgs.append(f"{op_name}: copy() lost epochs ({len(snapshot._data)})")
        else:
            for i, d in enumerate(snapshot._data):
                if d.shape != pre_lists[i].shape or not np.array_equal(
                    d, pre_lists[i]
                ):
                    msgs.append(
                        f"{op_name}: the copy taken beforehand was mutated at "
                        f"epoch {i}: shape {d.shape} vs {pre_lists[i].shape}"
                    )
        if len(snapshot.ch_names) != 4:
            msgs.append(
                f"{op_name}: copy() now has channels {snapshot.ch_names}"
            )
        # data list identity: contents replaced in place
        if id(ep._data) != pre_id:
            msgs.append(f"{op_name}: _data was rebound, not replaced in place")
        for i, d in enumerate(ep._data):
            if d.shape[-1] != lengths[i]:
                msgs.append(
                    f"{op_name}: epoch {i} lost samples: {d.shape[-1]} != "
                    f"{lengths[i]}"
                )
            if d.shape[0] != len(ep.ch_names):
                msgs.append(
                    f"{op_name}: epoch {i} has {d.shape[0]} channels, info says "
                    f"{len(ep.ch_names)}"
                )
        for m in msgs:
            fail(m)
        if not msgs:
            ok(
                f"{op_name}: ch={ep.ch_names} lens={lengths_of(ep)} "
                f"copy intact ({len(snapshot.ch_names)} ch)"
            )

    banner("caller's list must not be mutated by pick")
    # EpochsArray builds a new list but np.asarray keeps the caller's arrays
    from mne import create_info

    arrays = [np.zeros((4, L)) for L in lengths]
    for i, a in enumerate(arrays):
        a[0] = i * TAG_UNIT
    info = create_info([f"EEG{i:03d}" for i in range(4)], SFREQ, ["eeg"] * 4)
    stride = int(max(lengths)) + int(SFREQ) + 100
    events = np.column_stack(
        [np.arange(4) * stride + stride, np.zeros(4, int), np.ones(4, int)]
    )
    ep = mne.EpochsArray(
        arrays, info, events=events, tmin=np.zeros(4), event_id={"x": 1},
        baseline=None, verbose=False,
    )
    shapes_before = [a.shape for a in arrays]
    ep.pick(["EEG000", "EEG001"])
    shapes_after = [a.shape for a in arrays]
    if shapes_before != shapes_after:
        fail(f"pick mutated the caller's arrays: {shapes_before} -> {shapes_after}")
    else:
        ok(f"caller's arrays unchanged: {shapes_after}")


# ------------------------------------------------------------------- copy
def test_copy():
    banner("copy()")
    lengths = (100, 250, 75, 180)
    ep = make(lengths, (0.0, -0.2, 0.1, -0.5))
    cp = ep.copy()
    msgs = audit(cp, "copy()", expect_tags=[0, 1, 2, 3],
                 expect_tmins=(0.0, -0.2, 0.1, -0.5))
    if cp._tmin_per_epoch is ep._tmin_per_epoch:
        msgs.append("copy(): _tmin_per_epoch is the SAME array object")
    if cp._data is ep._data:
        msgs.append("copy(): _data is the SAME list object")
    for i in range(len(lengths)):
        if cp._data[i] is ep._data[i]:
            msgs.append(f"copy(): epoch {i} array is shared")
    if not cp._variable_duration:
        msgs.append("copy(): lost _variable_duration")
    # mutate the copy, check the original
    cp.shift_time(5.0)
    cp._data[0][:] = 99.0
    if not np.allclose(ep._tmin_per_epoch, (0.0, -0.2, 0.1, -0.5)):
        msgs.append(f"copy(): mutating the copy moved the original's bounds")
    if float(ep._data[0][0, 0]) != 0.0:
        msgs.append("copy(): mutating the copy's data changed the original")
    for m in msgs:
        fail(m)
    if not msgs:
        ok("copy() is independent in bounds and data")


# --------------------------------------------------- equalize_event_counts
def test_equalize():
    banner("equalize_event_counts")
    lengths = (100, 250, 75, 180, 120, 90)
    ep = make(lengths, event_ids=[1, 1, 1, 1, 2, 2])
    try:
        out, drop_idx = ep.equalize_event_counts(["e1", "e2"])
    except NotImplementedError as exc:
        ok(f"raises NotImplementedError: {str(exc)[:90]}")
        return
    except Exception as exc:
        fail(f"equalize_event_counts raised {type(exc).__name__}: {exc}")
        return
    print(f"       dropped indices {list(drop_idx)}")
    msgs = audit(out, "equalize_event_counts")
    kept = [t for t in range(len(lengths)) if t not in set(drop_idx)]
    if tags(out) != kept:
        msgs.append(f"tags {tags(out)} != expected kept {kept}")
    for m in msgs:
        fail(m)
    if not msgs:
        ok(
            f"tags={tags(out)} lens={lengths_of(out)} "
            f"tmin={np.round(out._tmin_per_epoch, 3)} sel={list(out.selection)}"
        )
    show(out, "after equalize")


# ------------------------------------------------------- concatenate_epochs
def test_concatenate():
    banner("concatenate_epochs")
    a = make((100, 250), (0.0, -0.2))
    b = make((75, 180), (0.1, -0.5))
    b._data[0][0] = 10 * TAG_UNIT
    b._data[1][0] = 11 * TAG_UNIT
    for label, args in [
        ("ragged + ragged", (a, b)),
    ]:
        try:
            out = mne.concatenate_epochs(list(args))
        except NotImplementedError as exc:
            ok(f"{label}: raises NotImplementedError: {str(exc)[:110]}")
            continue
        except Exception as exc:
            fail(f"{label}: raised {type(exc).__name__}: {exc}")
            continue
        msgs = audit(out, label)
        if tags(out) != [0, 1, 10, 11]:
            msgs.append(f"{label}: tags {tags(out)} != [0, 1, 10, 11]")
        if lengths_of(out) != [100, 250, 75, 180]:
            msgs.append(f"{label}: lengths {lengths_of(out)} != [100,250,75,180]")
        want_tmin = [0.0, -0.2, 0.1, -0.5]
        if out._tmin_per_epoch is not None and not np.allclose(
            out._tmin_per_epoch, want_tmin
        ):
            msgs.append(f"{label}: tmin {out._tmin_per_epoch} != {want_tmin}")
        for m in msgs:
            fail(m)
        if not msgs:
            ok(f"{label}: tags={tags(out)} lens={lengths_of(out)}")
        show(out, label)

    # ragged + fixed
    banner("concatenate_epochs: ragged + fixed")
    from mne import EpochsArray, create_info

    info = create_info(["EEG000", "EEG001", "EEG002"], SFREQ, ["eeg"] * 3)
    d = np.zeros((2, 3, 100))
    d[0, 0] = 20 * TAG_UNIT
    d[1, 0] = 21 * TAG_UNIT
    ev = np.column_stack([np.arange(2) * 400 + 400, np.zeros(2, int), np.ones(2, int)])
    fixed = EpochsArray(d, info, events=ev, tmin=0.0, event_id={"e1": 1},
                        baseline=None, verbose=False)
    a2 = make((100, 250), (0.0, -0.2))
    try:
        out = mne.concatenate_epochs([a2, fixed])
        print(f"       -> ok, variable={out._variable_duration}, "
              f"tags={tags(out) if isinstance(out._data, list) else 'ndarray'}")
    except Exception as exc:
        print(f"       -> {type(exc).__name__}: {str(exc)[:150]}")


# -------------------------------------------------------------- drop_bad
def test_drop_bad():
    banner("drop_bad with a reject threshold")
    lengths = (100, 250, 75, 180, 120)
    ep = make(lengths)
    # make epoch 1 and 3 exceed the threshold on channel 1
    ep._data[1][1, 0] = 500e-6
    ep._data[3][1, 0] = 500e-6
    try:
        ep.drop_bad(reject=dict(eeg=100e-6), verbose=False)
    except Exception as exc:
        fail(f"drop_bad raised {type(exc).__name__}: {exc}")
        return
    msgs = audit(ep, "drop_bad", expect_tags=[0, 2, 4])
    for m in msgs:
        fail(m)
    if not msgs:
        ok(
            f"tags={tags(ep)} lens={lengths_of(ep)} sel={list(ep.selection)} "
            f"drop_log={[d for d in ep.drop_log]}"
        )
    show(ep, "after drop_bad")


def main():
    test_shift_time()
    test_channels()
    test_copy()
    test_equalize()
    test_concatenate()
    test_drop_bad()
    banner("SUMMARY")
    print(f"{len(FAILS)} violations")
    for f in FAILS:
        print("  -", f)


if __name__ == "__main__":
    main()
