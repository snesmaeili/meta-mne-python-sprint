# Cross-post for #3533 / #5612 / #5794 / #11480 / #12315

> **Working notes, not text to paste.** MNE's contributing policy asks that AI-generated prose not be pasted into issue or PR descriptions. Use this as source material and write your own words. See `WRITING-NOTES.md`.

Short version of [`board-card.md`](board-card.md), sized for an issue comment.
Adjust the opening line per thread; keep the body identical so the threads
converge rather than fork again.

---

## Opening lines, per thread

**#3533** (closed WONTFIX) — "Reopening context rather than the issue: there is
now a working prototype and a benchmark for the container question this thread
closed on."

**#5612** — "This thread asked how to *analyse* variable-length epochs and was
repeatedly answered about how to *store* them. Here is a design that separates
the two, and a prototype that does the analysis part."

**#5794 / #11480** — "These fall out of the same design as the degenerate case
where durations happen to be equal. Details below; the prototype asserts
bit-identical behaviour with `mne.EpochsArray` in that case."

**#12315** — "@larsoner asked for a dev meeting on this in 2023. The 2026 sprint
is that meeting. Prototype and benchmark below, including an answer to
@agramfort's `nave` point, which I think is the objection that actually stopped
this PR."

---

## Body

For the sprint (board #10, "support for variable length epochs using
AwkwardArray?") I built a prototype and benchmarked the container question.
Everything below is runnable and requires no changes to mne-python:
`snesmaeili/meta-mne-python-sprint`.

**The framing I'd propose we adopt:** *raggedness is a data representation;
temporal alignment is an analysis decision.* Three problems keep getting
conflated — container (#3533, #12315), alignment (#5612), and per-trial time
origin (#5794, #11480). AwkwardArray addresses only the first. Separating them
is most of the work.

### 1. AwkwardArray does not earn its place — measured

2000 epochs × 128 channels @ 250 Hz, durations 0.8–1.7 s:

| backend | payload | random access | jagged reduce |
|---|---:|---:|---:|
| `list[np.ndarray]` + offsets | 587.4 MB | 1× | 1× |
| padded + lengths | 870.4 MB | 5.0× slower | 1.7× slower |
| `awkward.Array` | **587.4 MB** | **759× slower** | **12× slower** |

Awkward's payload is *byte-identical* to a plain list — both store exactly the
true samples. It saves memory versus padding, but so does a list, without a new
hard dependency. `reduce` is the per-epoch mean, i.e. the jagged reduction
awkward exists for; all three implementations agree to 1e-16.

Two structural points if anyone wants to revisit this:

- The only awkward type that enforces "only time is ragged" is
  `n_epochs * var * n_channels * float64`. The intuitive
  `ak.Array([b.T for b in blocks])` gives `n_epochs * var * var` — the channel
  dimension silently goes ragged too, with no error.
- That correct layout is `(epoch, time, channel)` while MNE is channel-major, so
  every dense conversion pays a transpose.

Scoped finding: for **one** ragged axis at EEG scale. If we ever want ragged
channels *and* ragged time (#11705 / #12219), the calculus changes.

### 2. The SciPy wall is still live — and it is not an argument for awkward

Verified on SciPy 1.17.1: `scipy.signal.spectrogram` given a
`np.ma.MaskedArray` returns a plain `ndarray`, mask silently dropped, masked
samples treated as real data. @alexrockhill's 2023 blocker stands.

But it applies identically to awkward arrays. The answer is architectural, not a
choice of container: **ragged for storage, indexing, metadata and IO; dense at
the point of computation, one epoch at a time.**

### 3. @agramfort's `nave` objection — I think this is the real one

From #12315, 2023-12-20: under padding the noise level becomes time-dependent,
`nave` becomes a function of time, and `nave` scales the noise covariance in the
inverse. That is a statistical objection, not an implementation detail, and I
don't think it's been answered.

The proposed answer: **no automatic reduction over ragged data.** `average()`
raises and names the alignment options. Aligned by warping, `nave` is constant
by construction. Aligned by padding, `pad()` *returns the `nave(t)` vector*
rather than hiding it behind a scalar. The time-dependence becomes visible
rather than silent.

### 4. Only 6 of 61 `Epochs` methods actually need a common time axis

@kingjr's Pandora's-box objection on #3533 is correct if you make each method
ragged-aware one at a time. Classified by mathematical meaning instead:

- 22 bookkeeping (no time axis involved)
- 12 spatial (channel axis only)
- 10 naturally ragged (per-trial; map over epochs)
- 4 length-changing (`crop`, `decimate`, `resample`, `shift_time`)
- 4 ragged-with-policy (`compute_psd` + the `plot_psd` family)
- 1 ragged-output (`compute_tfr(average=False)`)
- **6 requiring a common axis** (`average`, `standard_error`, `iter_evoked`,
  `plot_image`, `plot_topo_image`, `subtract_evoked`)
- 2 IO

**78% need no cross-trial decision at all.** Full table generated against the
installed MNE, so it stays honest as the API changes.

The policy row is the interesting one: `ICA.fit` already concatenates to
`(n_channels, n_epochs × n_times)`, so ragged ICA is natural — but should a 5 s
trial carry five times the influence of a 1 s trial? Same for covariance, and it
propagates into the inverse. Fixed-length epochs silently enforce
`sample weighting == epoch weighting`; ragged data forces the assumption into
the open. I'd call that a feature.

### 5. Where I'd start: honour `Annotations.duration`

`Epochs` has accepted `events=None → raw.annotations.onset` since 1.7 (#12311),
and the docstring says "the durations of the annotations are ignored in this
case". The ragged lengths are already in the data model and are being
deliberately discarded. Reading them is smaller and more reviewable than any new
constructor signature — and it's why #12315 was blocked on #12311 landing.

I would *not* propose `Epochs(raw, events, tmin=-0.2, tmax=[...])`. A per-epoch
array silently changes the return type and duplicates information we already
have.

One API opinion, which I think is the highest-risk decision here: **`times`
should raise** on ragged data rather than returning the shortest common
interval. A plausible-but-wrong time axis is the easiest way for this feature to
produce quiet scientific errors.

### 6. One methodological finding, in case it's useful regardless

EEGLAB's `newtimef` warps *after* the time-frequency transform, to the *median*
landmark latencies. Our own prior gait code warped the signal first. Two trials
carrying the same 10 Hz oscillation, at 1.0 s and 2.0 s:

```
native (no warping)          [ 10.0 Hz,  10.0 Hz]
path A  warp signal -> TFR   [ 10.0 Hz,  20.0 Hz]
path B  TFR -> warp TF axis  [ 10.0 Hz,  10.0 Hz]
```

Both give a common axis; only path B reports the right frequency. If MNE ever
ships time warping, the TF-domain version should be the one documented for
spectral work. Worth noting that in #5612 @AaronWill-Git independently arrived
at the TF-domain approach by instinct.

### On deconvolution

@kingjr's "use rERP" from #3533, in its modern form (`unfold`, Ehinger &
Dimigen 2019), is a real alternative and I don't want to talk past it.
Deconvolution solves overlap and continuous covariates; if trial duration is a
nuisance variable, it's the better tool. It does not give you the ERSP of a
variable-duration process aligned to its own internal landmarks — there's no
rERP formulation that puts toe-off at 12% of the gait cycle. Complementary,
not competing.

### And the honest caveat

Averaging time-warped ERSPs across subjects with different gait characteristics
can shift or reverse the phase of power changes relative to single-subject
results. Warping is not free, and the docs should say so.

---

Happy to take any of this apart. The parts I'm least sure about are the public
class name, whether `Epochs` with equal durations should *become* the ragged
class, and the FIF representation.
