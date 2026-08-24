# Skeleton for issue #14206

**Structure and facts. Write the sentences yourself** — MNE's policy is that
AI-generated prose should not be pasted into issue descriptions, and a reviewer
can tell. Aim for maybe 600–900 words; the detail lives in the linked repo.

Issue: https://github.com/mne-tools/mne-python/issues/14206 (currently empty)

Before writing:
- consider retitling; the current title prejudges the container question that
  the benchmark answers negatively
- re-add the five other assignees from the draft card: dnacombo, virvw,
  volerina, BelizSertcan, raphbrd

---

## Suggested sections

### Opening — 2–3 sentences

State what the issue is for. Points to hit:
- sprint board #10, converted from a draft card that had no description
- this consolidates #3533, #5612, #5794, #11480, #12315 rather than adding a
  sixth parallel thread — say this explicitly, it is the main risk
- there is a working prototype and a benchmark, links at the bottom

### The framing — short

One idea: storing trials of unequal length and deciding how to compare them are
different problems, and every previous thread merged them.

Evidence that they get merged:
- #3533 argued containers, got answered with "use rERP", which is an alignment
  answer
- #5612 says explicitly it is not about the container, gets pointed at #3533
- #12315 opened as a PSD entry point, turned into a container debate

Then the three-way split:
- container — #3533, #12315
- alignment — #5612
- per-trial time origin — #5794, #11480, which is the special case of alignment
  where durations happen to be equal

### Why now / who wants it

- #5612 open since 2018, mental arithmetic, the variable phase *is* the process
- @drammock on #12315: variable-length spoken sentences, keyword at varying
  latency, "recurrent on the forum"
- #5794: tone sequences with variable inter-onset intervals
- in #5612, @AaronWill-Git and @cbrnr independently converge in 2022 on the same
  workaround (longer fixed epochs → `EpochsTFR` → crop → interpolate)
- your own case: gait and table-tennis ERSP, had to reimplement epoching
  outside MNE. If you want one concrete line, the production code carries the
  feature request as a comment: swing cycles differ in duration by
  construction, so they cannot be stacked into one array at all

### On AwkwardArray specifically — this answers the card's question

Lead with the measurement, not the conclusion.

2000 epochs × 128 ch @ 250 Hz, durations 0.8–1.7 s:

| backend | payload | random access | jagged reduce |
|---|---:|---:|---:|
| `list[np.ndarray]` + offsets | 587.4 MB | 1× | 1× |
| padded + lengths | 870.4 MB | 5.0× slower | 1.7× slower |
| `awkward.Array` | 587.4 MB | 759× slower | 12× slower |

Points:
- payload byte-identical to a plain list; awkward saves memory against
  *padding*, but so does a list, without a new hard dependency
- `reduce` is the per-epoch mean — the jagged reduction awkward exists for; all
  three agree to 1e-16, so it is a fair comparison
- the layout trap: `ak.Array([b.T for b in blocks])` gives
  `n_epochs * var * var`, so the channel axis silently goes ragged too. The type
  that enforces the invariant is `n_epochs * var * n_channels`, which is
  time-major against MNE's channel-major convention
- scope the finding honestly: one ragged axis at EEG scale. Ragged channels
  *and* ragged time (#11705, #12219) would change the calculus

### Scope of the change — answers the Pandora's-box objection

@kingjr on #3533: supporting this means special-casing TFR, PSD, plotting,
`times`, covariance, ICA, `nave`, stats. True if done one method at a time.

Classified by mathematical meaning, `BaseEpochs`'s 61 public methods:
- 22 bookkeeping, 12 spatial, 10 naturally ragged, 4 length-changing
- 4 need a declared policy, 1 ragged output
- **6 actually need a common time axis**: `average`, `standard_error`,
  `iter_evoked`, `plot_image`, `plot_topo_image`, `subtract_evoked`
- 2 IO

48 of 61 need no cross-trial decision. Table is generated against the installed
MNE so it stays honest.

### The objections, answered by name

Keep these short, one or two sentences each.

- **@agramfort, #12315**: padding makes `nave` a function of time, which
  propagates into the noise covariance used by the inverse. Answer: no automatic
  reduction over ragged data; `average()` raises; `pad()` returns `nave` as a
  vector rather than a scalar. Say you think this is the objection that actually
  stopped that PR, not the SciPy one.
- **@alexrockhill, #12315**: SciPy drops `np.ma` masks. Still true — verified on
  SciPy 1.17.1. But it applies to awkward equally, so it argues for "dense at the
  computation boundary", not for a particular container.
- **@kingjr, #3533**, "just use rERP" / modern `unfold`: complementary. Give it
  real credit — deconvolution is better when duration is a nuisance variable. It
  does not give the ERSP of a variable-duration process aligned to its own
  internal landmarks.
- **@mmagnuski, #3533**, FieldTrip's variable-length trials were never that
  useful: fair. FieldTrip built the container and not the alignment layer, so
  users hit the wall one level later. Argues for layers 2–3, not against layer 1.
- **mne-connectivity #142** chose padding: correct there, no time axis. Cite as
  precedent for padding at the computation boundary.

### Where to start

- `Epochs` has accepted `events=None → raw.annotations.onset` since 1.7
  (#12311), and the docstring says the durations "are ignored in this case"
- so the ragged lengths are already in the data model and are being discarded
- that is the smallest reviewable first step, and it is why #12315 was blocked
  on #12311 landing
- say what you would *not* do: `Epochs(raw, events, tmin=-0.2, tmax=[...])`,
  because a per-epoch array silently changes the return type

### Open questions — ask, don't assert

This is what turns it into a discussion rather than a proposal to rubber-stamp.

1. Should `times` raise on ragged data, or return the shortest common interval?
   You think raise; a wrong time axis produces wrong results silently. Flag it as
   the highest-risk API decision.
2. New class, or extend `Epochs`? Prototype uses a new class to avoid changing a
   return type, and asserts the uniform case is bit-identical to `EpochsArray`,
   so they could merge later.
3. FIF: pad plus a lengths tag, or a new block type?
4. ICA and covariance weighting: sample or epoch? Note this is the interesting
   part — fixed-length epochs make the two identical and hide the choice.

### Links

- prototype + benchmark + method matrix: the sprint repo
- branch: `snesmaeili/mne-python` → `ragged-epochs`
- note the draft PR is coming and will reference this issue

### Note on @larsoner's 2023 request

He asked for a dev meeting on this on #12315 and never got one. Worth one line:
the sprint is that meeting, and this is the material for it.

---

## One methodological finding — include or hold?

Two trials, same 10 Hz oscillation, 1.0 s and 2.0 s:

```
native (no warping)          10.0 Hz,  10.0 Hz
warp signal -> TFR           10.0 Hz,  20.0 Hz
TFR -> warp TF axis          10.0 Hz,  10.0 Hz
```

EEGLAB `newtimef` warps after the transform, to the median landmark latencies.
This matters if MNE ever ships time warping, because the naive implementation is
the wrong one.

My suggestion: put this in the **PR**, not the issue. The issue is already doing
a lot, and this is an argument about the alignment layer rather than about
whether to support ragged epochs at all. If you do include it here, keep it to
the code block plus two sentences.

## Do not claim

- that the poster result validates `mne.ragged` — it came from
  `eneuro_merged_ersp.py`, and the reproduction script has not been run
- that 38.5% is Studnicki & Ferris's contact fraction — it is your cohort's
  pooled median; theirs is 33.3%
- that the per-trial operations are fast — they are a Python loop
- that awkward "does not work" — it works and loses on measurement
