# Variable-length / ragged epochs in MNE — issue archaeology

*Original sweep by @snesmaeili (`D:ariable-length-epochs-issues.md`), verbatim below. Section 5 adds three things the original sweep missed, found while building the prototype.*

Sprint item: **"support for variable length epochs using AwkwardArray?"**
(2026 MNE-Python Maintainers Sprint, board #10 — draft item, opened by @drammock,
status *Ideas*, assignees: dnacombo, virvw, volerina, BelizSertcan, raphbrd, snesmaeili.
No description on the card.)

Searched: `mne-tools/*` issues + PRs, 2011→2026, via the GitHub search API.
Terms: variable length / variable-length / variable duration / ragged / unequal /
different lengths / different durations / varying length / awkward / awkward array /
masked array / time warping / response-locked, plus cross-reference sweeps on #3533 and #12315.

---

## 0. The headline finding

**Nothing in any mne-tools repo has ever mentioned AwkwardArray.**
`org:mne-tools awkwardarray` → 0 hits. `org:mne-tools "awkward array"` → 1 hit, and it is a
false positive (mne-connectivity #142, about ragged *connectivity seeds/targets*, not the library).
The ~47 mne-python hits for `awkward` are all the ordinary English word.

So the library proposal is new. What is *not* new is the underlying feature request — it has a
10-year paper trail, one WONTFIX, two still-open issues, and one stalled PR.

---

## 1. The core thread (read these four)

### #3533 — ENH: Variable duration epochs
*2016-08-26 · @larsoner · **closed WONTFIX** 2017-02-14 · 13 comments*
<https://github.com/mne-tools/mne-python/issues/3533>

The canonical design discussion. Motivating case: sentence stimuli, so each trial has its own
`tmax`; `tmax=min(all_tmax)` throws away data, `tmax=max(all_tmax)` gives `TOO_SHORT` at run ends.

Two options put on the table:
1. **True variable `tmax`** — dismissed as too hard given the IO and class structure.
2. **`too_short='omit' | 'nan'`** — pad the missing tail with `np.nan`, let `nan` propagate,
   add a `_check_finite` guard to functions that can't cope.

Positions: @larsoner for (2). @kingjr **-1**, the "Pandora's box" argument — TFR, PSD, plotting,
`times`, covariance, ICA, `nave`, stats all need NaN special-cases; argues an Epochs container is
*by definition* for comparable repetitions; suggests rERP instead. @jona-sassenhagen initially agreed
with kingjr despite doing variable-length work himself (sentences/stories), then softened.
@mmagnuski preferred (2) but noted FieldTrip's variable-length trials were never that useful in practice.
@christianbrodbeck floated using `Raw` as the container. @agramfort: same first reaction as kingjr,
but "eager to see a draft of solution 2".

**This is the issue any new proposal has to answer.** The objections are still live.

### #5612 — Compute t/f representation with variable-length epochs?
*2018-10-16 · @cbrnr · **open** · 41 comments — the longest discussion in the set*
<https://github.com/mne-tools/mne-python/issues/5612>

Different framing: not "how do we store them" but "how do we *analyse* them". Mental-arithmetic
task, problem-solving phase varies per trial and per subject. Explicitly says it is **not** about
the container (defers to #3533).

Key point: NaN-padding does **not** solve the analysis problem, because averaging TFRs across
trials of unequal length averages misaligned processes. What's wanted is **time warping** —
EEGLAB's `newtimef` timewarp — stretching each trial to 100% of its own process before averaging.
Workarounds discussed: Hilbert envelope on `Raw`, binning trials by length (@agramfort),
masking the invalid TF region.

### #5794 — epochs realignment: shifting time by a variable amount per epoch
*2018-12-15 · @kdoelling1919 · **open** · 13 comments*
<https://github.com/mne-tools/mne-python/issues/5794>

Third framing: keep the epochs rectangular, but let *time zero* differ per epoch. Tone sequences
with variable inter-onset intervals; wants baseline locked to first tone but analysis re-locked to
the *n*th tone. `epochs.shift_time` can't do per-epoch shifts. Proposes a `TimeMixin` shared by
`Epochs`, `EpochsTFR`, `SourceEstimate`, and moving `shift_time`/`crop` into it.

### #12315 — [ENH] PSD of Ragged Epochs
*2023-12-19 · @alexrockhill · **open draft PR** · 11 comments — the most recent state of the art*
<https://github.com/mne-tools/mne-python/pull/12315>

Starts bottom-up (`raw.compute_psd(..., ragged_epochs=True)` using `annotations.duration`), and the
discussion turns it top-down. The substance:

- **@larsoner**: fix `Epochs` *first*, downstream follows; a bottom-up entry point will just have to
  be deprecated in favour of `Epochs(raw, ..., allow_ragged=True).compute_psd()`. Proposes keeping
  the ndarray of shape `(n_epochs, n_channels, n_times_max)` but making it a
  **`np.ma.masked_array`**. Suggests a dev meeting on the details.
- **@larsoner** again, on the bottom-up route: you end up reimplementing `Epochs` without `Epochs`
  — losing `reject`, `flat`, `reject_by_annotation`, edge handling, `drop_log` — and then you've
  arrived at variable-length `Epochs` the long way round.
- **@drammock**: if we support *any* variable-length analysis in module code, it should address the
  **general** case. Gives the concrete use case: variable-length spoken sentences, target keyword at
  varying latency, baseline pre-sentence-onset — common in language processing, and recurrent on the
  forum. Even without correct statistics it buys you epoching, baselining, and inverse application.
- **@alexrockhill** started -0.5 on general ragged support, was convinced, tried masked arrays, and
  hit the wall: **SciPy does not handle `np.ma` masked arrays** (`spectrogram` ignores the mask).
  `evoked.plot`, `epochs.plot`, `epochs.save`/`read_epochs` did work.

That SciPy wall is arguably the strongest argument for the AwkwardArray framing — but note it cuts
both ways: SciPy won't consume an awkward array either.

---

## 2. Enabling infrastructure (already merged)

| # | Date | State | Why it matters |
|---|---|---|---|
| [#12311](https://github.com/mne-tools/mne-python/pull/12311) | merged 2023-12-31 | ✅ | **Allow epoch construction from annotations.** This is the natural entry point — `Annotations` already carry per-event `duration`, so the ragged lengths are already in the data model. #12315 was explicitly blocked on this landing. |
| [#11465](https://github.com/mne-tools/mne-python/pull/11465) | 2023-02-10 | ✅ | Object-array support — precedent for non-rectangular data in the codebase. |
| [#11803](https://github.com/mne-tools/mne-python/pull/11803) | 2023-07-13 | ✅ | `EpochsSpectrumArray` / `SpectrumArray` — the classes a ragged PSD would have to produce. |
| [#9969](https://github.com/mne-tools/mne-python/pull/9969) | 2021-11-05 | ✅ | Epochs store `Annotations` from `Raw`. |
| [#3993](https://github.com/mne-tools/mne-python/pull/3993) | 2017-02-15 | ✅ | Annotation-aware data getter. |

## 3. Adjacent, worth citing

- [#11480](https://github.com/mne-tools/mne-python/issues/11480) (2023-02-15, **open**) — allow the
  baseline window to come from a *different* event. Same family as #5794: per-trial time structure
  that the rectangular model can't express.
- [#11705](https://github.com/mne-tools/mne-python/issues/11705) (2023-05-23, **open**) +
  [#12219](https://github.com/mne-tools/mne-python/pull/12219) (**open**) — channel-specific epoch
  removal. Raggedness along the *channel* axis rather than the time axis; a masked-array or awkward
  representation would address both at once. Good ammunition for the "general case" argument.
- [#1963](https://github.com/mne-tools/mne-python/issues/1963) (2015-04-13) — MNE events
  functionality; early events-API rethink, mentions variable length.
- [#1016](https://github.com/mne-tools/mne-python/issues/1016) (2013-12-26) — single-trial
  regression, and [#2304](https://github.com/mne-tools/mne-python/pull/2304) rERP/rERF: the
  "just use rERP" counter-proposal from #3533.
- [#119](https://github.com/mne-tools/mne-python/issues/119) (2012-09-26) — earliest hit in the
  whole sweep; epoch-dropping metadata.
- [#10367](https://github.com/mne-tools/mne-python/issues/10367) (2022-02-21, **open**) — time
  x-axis labels in `epochs.plot`; the viz side of "epochs don't share a time axis".
- mne-connectivity [#142](https://github.com/mne-tools/mne-connectivity/pull/142) (2023-07-24) —
  ragged multivariate connections, solved by **padding**. The only place the ecosystem has actually
  shipped a ragged-data decision, and it went with padding, not a ragged container.

---

## 4. What the history says for the sprint

1. **Three distinct problems keep getting conflated.** (a) *Container*: how to store trials of
   unequal length (#3533, #12315). (b) *Alignment*: how to average processes of unequal duration —
   time warping (#5612). (c) *Per-trial time origin*: rectangular data, different t=0 (#5794, #11480).
   AwkwardArray only addresses (a). Saying so explicitly up front would keep the thread from
   relitigating #3533.
2. **The container debate has already converged once**, on masked arrays of shape
   `(n_epochs, n_ch, n_times_max)` (#12315). A proposal should say why AwkwardArray beats
   `np.ma` — and the honest answer is probably memory for very unequal lengths, plus a real
   ragged-aware API, versus the cost of a new hard dependency.
3. **The killer objection is downstream, not upstream.** #12315 died on SciPy ignoring masks.
   Awkward arrays have the same problem. Any credible plan needs a story for the
   ndarray boundary — likely "convert to padded dense at the point of computation, stay ragged
   for storage/IO/indexing".
4. **@larsoner asked for a dev meeting on this in 2023** and never got one. The sprint is that meeting.
5. **The demand is real and repeated**: larsoner's lab (3+ times by 2016), cbrnr, kdoelling1919,
   jona-sassenhagen, alexrockhill, drammock (plus forum questions drammock recalls). Kingjr's
   "rare use case" claim was contested in 2016 and has aged poorly.

---

## 5. Additions from the prototype work (2026-08)

Three things the original sweep did not capture. Found by reading the issue
threads to the end and by checking the claims against a current environment.

### 5.1 The objection that actually stopped #12315 was statistical, not technical

Section 4.3 above credits the SciPy masked-array wall. That was real, but it was
not the last word. @agramfort, 2023-12-20, replying to @alexrockhill's working
masked-array demo:

> "Yes it works numerically but I would question how much it works
> statistically. The noise level then becomes time dependent. Your `nave`
> becomes a function of time so the N= you see in plots is misleading. Also see
> how `nave` is used to scale the noise covariance in the inverse problem.
> Basically I would suggest to think beyond an implementation issue here."

@alexrockhill's reply — "probably something better as a discussion than a reply
thread" — is the last substantive comment on the PR. It has not been touched
since.

This matters for how a new proposal is pitched. A container proposal that only
answers the SciPy wall answers the *second*-hardest objection. `nave` becoming a
function of time is a property of **padding**, not of raggedness — which is why
the architecture forbids automatic reduction over ragged data and, when padding
is chosen explicitly, returns the `nave(t)` vector rather than a scalar.

### 5.2 Independent convergence on the same workaround

Section 4.5 lists demand by maintainer name. Stronger evidence sits inside
#5612: a 2022 exchange between @AaronWill-Git and @cbrnr in which a gait-EEG
user, with no connection to this work, independently arrives at exactly the
mne-mobi workaround —

> "My plan is to divide each step into fixed epochs with a longer length than
> the normal step. Then I calculate the ERSP using
> `mne.time_frequency.EpochsTFR`. Next step I want to interpolate the results of
> EpochsTFR [...] crop to the target step length and linearly interpolate it to
> a longer fixed length."

@cbrnr's answer: "You can create a new `TFREpochs` object and fill it with the
new data."

Two groups reaching the same workaround independently is better evidence of a
missing abstraction than another maintainer +1. It also confirms that users get
the TF-domain interpolation right by instinct — which makes it more, not less,
important that the library not ship the signal-domain version as the default.

### 5.3 Checked and ruled out, so it does not get raised as a dependency

- **#13927 "GSoC 2026: Event System meta issue"** (open, last updated
  2026-08-20) is about **bidirectional UI/plot event propagation** between
  figures — `TimeChange`, `ChannelsSelection`, `VertexSelect`. It is not about
  events-as-annotations and is not a dependency. The name invites confusion.
- **No post-2024 mne-python issue or PR revisits ragged epochs.** #12315 last
  touched 2023-12-20; #5612 last touched 2024-05-17. A search of
  `repo:mne-tools/mne-python` for ragged / variable length / variable duration /
  awkward since 2024-01-01 returns nothing on topic. The field is clear.
- **The SciPy wall is still live.** Verified on SciPy 1.17.1 (2026-08):
  `scipy.signal.spectrogram` given a `np.ma.MaskedArray` returns a plain
  `ndarray` with the mask silently dropped and masked samples treated as real
  data. Not inherited knowledge from 2023 — measured now.
- **`mne.Epochs` still discards annotation durations.** Docstring, MNE 1.11:
  "If `raw` contains annotations, `Epochs` can be constructed around
  `raw.annotations.onset`, but note that the durations of the annotations are
  ignored in this case." This is the cheapest entry point and it is still open.
