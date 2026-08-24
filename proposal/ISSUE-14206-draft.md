# Draft body for #14206 — rewrite before posting

**Retitle to:** `Support variable-duration epochs`
**Assignees to restore:** dnacombo, virvw, volerina, BelizSertcan, raphbrd

Links are filled in. Repo is public at
<https://github.com/snesmaeili/meta-mne-python-sprint>.

---

This is the board #10 sprint item, converted to an issue. I'd like it to be one
place to work through a request that has come back repeatedly since 2016 rather
than a sixth parallel thread: #3533, #5612, #5794, #11480, #12315.

Thanks @larsoner for the sketch — replying to it below. I also built a prototype
to test the design space ([repo](https://github.com/snesmaeili/meta-mne-python-sprint), branch
[`ragged-epochs`](https://github.com/snesmaeili/mne-python/tree/ragged-epochs)).
It's a probe, not something I'm proposing to merge.

### Three problems that keep getting mixed together

- **Representation** — storing trials of unequal length. #3533, #12315.
- **Within-trial temporal anchors** — #5794 wants a different zero point per
  trial; #11480 wants a baseline defined by a *different* event in the same
  trial. Neither needs variable durations, both need more than one landmark per
  trial.
- **Alignment and reduction** — how to compare processes of unequal duration.
  #5612.

The threads keep crossing between these even when the original request concerns
only one. #3533 asks for a container and gets answered with rERP. #5612 says
explicitly that it is *not* about the container and gets pointed at #3533.

### Who wants it

- #5612, open since 2018: mental arithmetic, where the variable-duration
  problem-solving period is the thing being studied. Truncating it to a common
  window changes the question.
- @drammock on #12315: variable-length spoken sentences, keyword at varying
  latency, baseline before sentence onset.
- #5794: tone sequences with variable inter-onset intervals.
- In #5612 in 2022, a gait user and the original author converge independently on
  the same workaround — epoch to a longer fixed window, compute `EpochsTFR`, crop
  to the true duration, interpolate the TF time axis.

I ran into the same thing in a mobile EEG pipeline. Gait cycles have unequal
durations, so each cycle had to be pulled out of `Raw` separately and analysed on
its own before anything rectangular could be built. That's machinery that belongs
near the epoch abstraction rather than in every downstream project.

### On the sketch

**Agreed on ndarray `tmin`/`tmax` over a separate class.** I prototyped a
separate class first and I think it was the wrong call — it means duplicating or
inheriting 61 public methods, and my objection (that array `tmax` changes what
the constructor returns) doesn't hold up, since arrays are opt-in at
construction.

One migration question: once `epochs.tmin` can be an array, code that receives an
`Epochs` from elsewhere and does scalar arithmetic on `.tmin`/`.tmax` silently
changes behaviour. Should scalar access stay valid in the uniform case?

**On `awkward` as an optional dep.** If it's soft-imported there has to be a
fallback that works without it, so I benchmarked three backends — list of arrays
plus offsets, padded plus lengths, and awkward — at 2000 epochs × 128 channels,
250 Hz, durations 0.8–1.7 s
([numbers and script](https://github.com/snesmaeili/meta-mne-python-sprint/blob/main/docs/06-container-benchmark.md)).

The list and awkward both store only the observed samples, with effectively equal
data-buffer payload. Padding costs about 48% more, since it materialises
`max(n_times)` for every epoch. On every access and reduction pattern I tested,
including the jagged reduction awkward exists for, the list was substantially
faster.

So on this evidence I'm not sure the optional dependency buys anything. But I
didn't measure serialisation, memory-mapping, or interop with a shared ecosystem
representation, which are the obvious reasons to want it — is that what you had
in mind?

One implementation note if awkward is used: `ak.Array([b.T for b in blocks])`
infers `n_epochs * var * var`, so the channel axis becomes variable too. Keeping
channels regular needs `n_epochs * var * n_channels`, which is time-major against
MNE's channel-major convention.

**On `warn` + `as_fixed()`.** I think this is right for most methods and needs a
carve-out for a few. `as_fixed()` with `tmin=min, tmax=max` spans the union
window, so shorter epochs get filled. That's padding, and it's what @agramfort
objected to on #12315: the noise level becomes time-dependent, `nave` becomes a
function of time, and `nave` scales the noise covariance used by the inverse. A
warning that a method "may not behave correctly" doesn't convey that.

I went through all 61 public `BaseEpochs` methods and classified them by whether
they're mathematically defined by a reduction across a shared time axis. Only six
are: `average`, `standard_error`, `iter_evoked`, `subtract_evoked`,
`plot_image`, `plot_topo_image`
([full table](https://github.com/snesmaeili/meta-mne-python-sprint/blob/main/docs/05-method-matrix.md)). The rest are bookkeeping, spatial,
per-trial or length-changing, and wrapping those looks cheap. So the wrapper
could apply broadly, with those six raising or taking an explicit policy.

### The part the sketch doesn't cover

All four points are about the container and the migration path. The harder half
is what happens once unequal-duration trials have to be compared, and it has a
default that looks right and isn't. Two trials carrying the same 10 Hz
oscillation, at 1.0 s and 2.0 s:

```
native (no warping)          10.0 Hz,  10.0 Hz
warp signal -> TFR           10.0 Hz,  20.0 Hz
TFR -> warp TF axis          10.0 Hz,  10.0 Hz
```

Warping the signal before the transform rescales the frequencies it carries, and
it's the implementation you reach for first. EEGLAB's `newtimef` warps the
representation instead, to the median landmark latencies. Worth deciding
deliberately rather than leaving to follow-ups.

### A cheap starting point

#12311 already lets `Epochs` build events from `raw.annotations.onset`, and
explicitly logs that annotation durations are ignored. The durations are already
in `Raw.annotations` and are discarded on the way in. Reading them fits the
ndarray-`tmax` direction rather than competing with it.

### Open questions

1. Should scalar `.tmin`/`.tmax` access stay valid in the uniform case?
2. What should reductions do when the contributor count varies over time?
   `Evoked.nave` is scalar. Related: with fixed lengths, weighting ICA and
   covariance by sample and by epoch coincide; with variable lengths they're
   different estimators.
3. FIF — pad plus length metadata, or a representation specific to
   variable-length trials?

#12315 proposed a dev meeting to work through the details. The sprint seems like
a good opportunity for that.

---

## Checks before posting

- [ ] rewritten in your own words
- [x] links filled and resolving
- [ ] retitled, assignees restored
- [ ] you can defend the six-method list and the benchmark if asked
- [ ] no claim that the poster validates any of this
- [ ] 38.5% not attributed to Studnicki & Ferris (that's 33.3%; 38.5% is your
      cohort's pooled median)
