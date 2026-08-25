"""Does the variable-duration container hold up on sleep_physionet?

Every claim the tutorial will make, checked against SC4001 before any of it is
written into a doc.
"""

import numpy as np

import mne
from mne.datasets.sleep_physionet.age import fetch_data

mne.set_log_level("WARNING")

(psg, hyp), = fetch_data(subjects=[0], recording=[1], verbose=False)
raw = mne.io.read_raw_edf(psg, stim_channel=False, preload=False, verbose=False)
raw.pick(["EEG Fpz-Cz", "EEG Pz-Oz"])
annot = mne.read_annotations(hyp)
raw.set_annotations(annot, emit_warning=False)
sfreq = raw.info["sfreq"]

STAGES = {
    "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}

onset, dur, desc = annot.onset, annot.duration, np.array(annot.description)
keep = np.array([d in STAGES for d in desc])
# stay inside the recording
keep &= (onset + dur) <= (raw.times[-1] + 1.0 / sfreq)

onset, dur, desc = onset[keep], dur[keep], desc[keep]
codes = np.array([STAGES[d] for d in desc])
events = np.column_stack([
    np.round((onset - raw.first_time) * sfreq).astype(int),
    np.zeros(len(onset), int),
    codes,
])
event_id = {"N1": 1, "N2": 2, "N3/4": 3, "REM": 4}

print(f"bouts kept : {len(events)}")
print(f"durations  : {dur.min():.0f}-{dur.max():.0f} s, median {np.median(dur):.0f} s")

# ---- the thing under test --------------------------------------------------
epochs = mne.Epochs(
    raw, events, event_id,
    tmin=np.zeros(len(events)),
    tmax=dur - 1.0 / sfreq,     # inclusive endpoint
    baseline=None, preload=True, verbose=False,
)

print(f"\nvariable_duration : {epochs.variable_duration}")
print(f"len               : {len(epochs)}")

# C1: durations round-trip against the annotations
got = epochs.durations
assert np.allclose(got, dur, atol=1.5 / sfreq), (got[:5], dur[:5])
print(f"C1 durations match annotations   OK   ({got.min():.0f}-{got.max():.0f} s)")

# C2: each epoch is byte-identical to the raw slice it came from
data = epochs.get_data()
bad = 0
for i in range(len(epochs)):
    s = events[i, 0]
    want = raw.get_data(start=s, stop=s + data[i].shape[-1])
    if not np.array_equal(data[i], want):
        bad += 1
print(f"C2 every epoch == its raw slice  {'OK' if bad == 0 else f'{bad} MISMATCH'}"
      f"   ({len(epochs)} epochs, no padding)")

# C3: get_times is per-epoch and the right length
ok = all(len(epochs.get_times(i)) == data[i].shape[-1] for i in range(len(epochs)))
print(f"C3 get_times(i) matches n_times  {'OK' if ok else 'FAIL'}")

# C4: native ops keep durations
sub = epochs["N2"]
print(f"C4 epochs['N2']                  OK   n={len(sub)}, "
      f"{sub.durations.min():.0f}-{sub.durations.max():.0f} s")
picked = epochs.copy().pick(["EEG Pz-Oz"])
assert np.allclose(picked.durations, epochs.durations)
print(f"C5 pick keeps durations          OK   {picked.ch_names}")

# C6: the ambiguous ones refuse
for name, fn in [("times", lambda: epochs.times),
                 ("average()", lambda: epochs.average())]:
    try:
        fn()
        print(f"C6 {name:<12} did NOT refuse   FAIL")
    except Exception as e:
        print(f"C6 {name:<12} refuses          OK   {type(e).__name__}")

# C7: as_fixed makes padding explicit
fixed, n_contrib = epochs.as_fixed()
waste = 100 * (1 - epochs.durations.sum() / (epochs.durations.max() * len(epochs)))
print(f"C7 as_fixed()                    OK   {fixed.get_data().shape}, "
      f"n_contributing {n_contrib.max()} -> {n_contrib.min()}")
print(f"   padding waste                      {waste:.1f}%")
print(f"   support halves by                  "
      f"{np.searchsorted(-n_contrib, -n_contrib.max() / 2) / sfreq:.0f} s")
