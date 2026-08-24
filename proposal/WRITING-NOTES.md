# Notes for writing the issue and PR

**Do not paste prose from this file, or from `board-card.md` / `issue-comment.md`,
into GitHub.** MNE's contributing policy: *"Please do not paste AI generated text
in the description of issues, PRs or in comments."* These are facts, numbers and
argument structure for you to write from in your own words.

Branch: `snesmaeili/mne-python` → `ragged-epochs` (commit `db9b6f5`, pushed).
Open a PR from: https://github.com/snesmaeili/mne-python/pull/new/ragged-epochs

---

## 1. What is on the branch

14 files, +2586 lines, no deletions.

| File | What it is |
|---|---|
| `mne/ragged/_container.py` | `RaggedEpochs`, `RaggedTimesError` |
| `mne/ragged/_align.py` | crop / pad / duration-normalize / landmark warp |
| `mne/ragged/_tfr.py` | `RaggedEpochsTFR`, `compute_tfr`, `warp_tfr` |
| `mne/ragged/_ops.py` | per-trial filter, baseline, reference, covariance |
| `mne/ragged/_provenance.py` | `AlignmentRecord` |
| `mne/ragged/tests/test_ragged.py` | 28 tests |
| `doc/api/ragged.rst` + toctree entry | API page |
| `doc/changes/dev/14210.newfeature.rst` | **rename to the real PR number after opening** |
| `doc/changes/names.inc` | your contributor entry |
| `mne/__init__.pyi` | lazy-load registration |
| `mne/tests/test_docstring_parameters.py` | registers `mne.ragged` for docstring CI |

Checks run locally, all green:

```
python -m pytest mne/ragged mne/tests/test_docstring_parameters.py mne/tests/test_epochs.py
284 passed, 10 skipped, 3 xfailed
ruff check mne/ragged/   -> All checks passed
ruff format --check      -> 6 files already formatted
```

The 40 `ruff check mne/ doc/` errors are pre-existing upstream, none in `mne/ragged`.

---

## 2. Design decisions you need to be able to defend

Reviewers will ask about these. Each is a real judgment call with a real
alternative, so be ready to say why.

**`times` raises instead of returning the shortest common interval.**
Alternative: return something plausible so existing code keeps running. Reason
against: a wrong time axis produces wrong results that never surface as an
error. The message names `durations`, `get_times()` and `align_time()`. This is
the decision most likely to be challenged, and reasonable people could disagree.

**New class rather than extending `Epochs`.** Alternative: make
`mne.Epochs` return ragged data when annotation durations vary. Reason against:
silently changing a return type. Cost of the choice: two classes to maintain.
Mitigation: `RaggedEpochs` with equal durations is asserted bit-identical to
`EpochsArray` (`test_uniform_case_matches_epochs_array`), so they could be
merged later.

**No AwkwardArray, despite the board card naming it.** Numbers below. Be ready
for "did you configure it correctly" — the answer is that the naive construction
is wrong and you know why (see §3).

**`sfreq` stays physical.** Alternative: set it to `n_points - 1` for a percent
axis, which is what our own earlier gait code did. Reason against: after that,
600 ms and 60% of a trial are the same number in the same field, and the
original durations are gone. Pinned by `test_sfreq_is_never_a_phase_axis`.

**Warp target defaults to the median.** Matches EEGLAB `newtimef`
(`timerefs = median(...)`). `'uniform'` exists but must be asked for by name;
for gait it puts toe-off at 25% of the cycle instead of ~12%.

**`pad()` returns `nave` as a vector.** Directly answers agramfort's objection on
#12315. Be able to state that objection accurately: under padding the noise
level becomes time-dependent, `nave` becomes a function of time, the `N=` shown
in plots misleads, and `nave` scales the noise covariance in the inverse.

**Context padding.** Adopted from our own production pipeline. Without it, 10 Hz
power at the epoch edge measures 0.37 of its mid-epoch value; with 1 s of
context, 0.99. Those edges are the cycle-start and cycle-end landmarks.

**DTW deliberately excluded from v1.** It derives correspondence from signal
similarity rather than from known events, so it can align noise and change
apparent component durations.

---

## 3. Numbers to use

**Frequency preservation.** Two trials, same 10 Hz oscillation, 1.0 s and 2.0 s:

```
native (no warping)          10.0 Hz,  10.0 Hz
warp signal -> TFR           10.0 Hz,  20.0 Hz
TFR -> warp TF axis          10.0 Hz,  10.0 Hz
```

**Container benchmark.** 2000 epochs × 128 channels @ 250 Hz, 0.8–1.7 s:

| backend | payload | random access | jagged reduce |
|---|---:|---:|---:|
| `list[np.ndarray]` + offsets | 587.4 MB | 1× | 1× |
| padded + lengths | 870.4 MB | 5.0× slower | 1.7× slower |
| `awkward.Array` | 587.4 MB | 759× slower | 12× slower |

`reduce` = per-epoch mean, the jagged reduction awkward exists for; all three
agree to 1e-16. Awkward payload is byte-identical to the list.

**Awkward layout trap** (worth stating, it is not obvious):
`ak.Array([b.T for b in blocks])` gives `n_epochs * var * var` — the channel
axis silently becomes ragged too. The type that enforces the invariant is
`n_epochs * var * n_channels`, which is `(epoch, time, channel)` while MNE is
channel-major, so every dense conversion pays a transpose.

**API scope.** `BaseEpochs` has 61 public methods. 22 bookkeeping, 12 spatial,
10 naturally ragged, 4 length-changing, 4 need a declared policy, 1 ragged
output, 6 need a common axis, 2 IO. 48 of 61 need no cross-trial decision.

**SciPy masks.** Verified on SciPy 1.17.1: `scipy.signal.spectrogram` on a
`np.ma.MaskedArray` returns a plain `ndarray`, mask dropped, padded samples
treated as real data. Applies to awkward equally, so it is not an argument for
one container over another.

**Demand.** #5612 (mental arithmetic, still open), @drammock on #12315
(variable-length spoken sentences), #5794 (tone sequences). In #5612,
@AaronWill-Git and @cbrnr converge in 2022 on the same workaround: longer fixed
epochs → `EpochsTFR` → crop → interpolate.

---

## 4. Sequence

1. **Issue first.** Frame it as consolidating #3533 / #5612 / #5794 / #11480 /
   #12315, and say so in the first line — a sixth thread needs to justify itself.
2. **PR second**, as a **draft**, referencing the issue. #12315 opened code
   before the design was agreed and became a design debate; the draft flag and
   the issue link are what avoid repeating that.
3. **Rename** `doc/changes/dev/14210.newfeature.rst` to the real PR number.
   MNE's changelog bot has `verify_pr_number = true`.
4. Short pointers on #5612 and #12315 so those threads converge on the issue.

---

## 5. Disclosure

Required by CONTRIBUTING.md, in the PR description. Their own examples are of
the form *"I implemented the code changes myself, and Claude Sonnet 4.6 wrote
the test."* Write your own version; it needs to be accurate about tool, manner
and scope. What actually happened:

- The analyses this is based on are yours: the cluster ERSP runs, the poster,
  the ds004505 replication.
- The architecture was worked out in conversation with Claude Opus 4.6.
- Claude wrote the prototype, the ported `mne/ragged` code, and the tests.
- You reviewed and tested it before submitting.

That last point has to be true before you submit. Read the branch.

---

## 6. Things I would not claim in the PR

- Do not say the poster result validates this code. The poster came from
  `eneuro_merged_ersp.py`, not from `mne.ragged`. The reproduction script
  (`validation/reproduce_poster_ersp.py` in the sprint repo) has not been run.
- Do not present 38.5% as Studnicki & Ferris's value. It is our cohort's pooled
  median; theirs is 33.3%.
- Do not claim the layer-2 operations are fast. They are a Python loop over
  epochs, which is why the docstring calls them the reference implementation.
- Do not say awkward "does not work". It works; it loses on measurements at
  this data shape, and the finding is scoped to one ragged axis at EEG scale.
