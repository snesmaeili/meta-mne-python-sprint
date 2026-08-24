# Draft body for #14206

Title is done: **Support variable-duration epochs**.
Assignees can't be set without admin rights. The closing line asks @drammock to
restore them (he opened the board card) and tags the five in the meantime.

All links resolve. **Read it through and make it yours before posting** — and put
the AI-assistance disclosure on the PR when you open it, per CONTRIBUTING.md.

---

Sprint board #10, converted to an issue. I'd like to use it to work through a
request that keeps coming back — #3533, #5612, #5794, #11480, #12315 — rather
than start a sixth thread alongside them.

Thanks @larsoner for the sketch. Replying to it below. I also put together a
prototype to test the design space:
[repo](https://github.com/snesmaeili/meta-mne-python-sprint), branch
[`ragged-epochs`](https://github.com/snesmaeili/mne-python/tree/ragged-epochs).
It's a probe, not a merge proposal.

### Three problems that keep getting mixed together

**Representation.** Storing trials of unequal length. #3533, #12315.

**Within-trial temporal anchors.** #5794 wants a different zero point per trial.
#11480 wants a baseline defined by a *different* event in the same trial. Neither
needs variable durations. Both need more than one landmark per trial.

**Alignment and reduction.** How to compare processes of unequal duration. #5612.

The threads cross between these even when the original request is about one of
them. #3533 asks for a container and gets answered with rERP. #5612 says
explicitly that it isn't about the container, and gets pointed at #3533.

### Who wants it

#5612 has been open since 2018. It's a mental arithmetic task, and the
variable-duration solving period is the thing being studied — cutting every trial
to a common window changes the question.

@drammock raised variable-length spoken sentences on #12315: keyword at varying
latency, baseline before sentence onset. #5794 is tone sequences with variable
inter-onset intervals.

In 2022 a gait user turned up in #5612 and converged, independently, on the same
workaround as the original poster: epoch to a longer fixed window, compute
`EpochsTFR`, crop to the true duration, interpolate the TF time axis.

I hit it in a mobile EEG pipeline. Gait cycles have unequal durations, so every
cycle had to come out of `Raw` separately and be analysed on its own before
anything rectangular could be built. That's a fair amount of machinery to
reimplement per project.

### On the sketch

**ndarray `tmin`/`tmax` over a separate class — agreed.** I prototyped a separate
class first and I now think that was wrong. It means duplicating or inheriting 61
public methods. And my objection to array `tmax` (that it changes what the
constructor returns) doesn't survive contact, since you only get arrays if you
pass arrays.

One migration question. Once `epochs.tmin` can be an array, code that gets an
`Epochs` from somewhere else and does scalar arithmetic on `.tmin` or `.tmax`
changes behaviour silently. Should scalar access stay valid in the uniform case?

**`awkward` as an optional dep.** If it's soft-imported there has to be a
fallback that works without it, so I benchmarked three: list of arrays plus
offsets, padded plus lengths, and awkward. 2000 epochs × 128 channels, 250 Hz,
durations 0.8–1.7 s.
[Numbers and script.](https://github.com/snesmaeili/meta-mne-python-sprint/blob/main/docs/06-container-benchmark.md)

List and awkward both store only the observed samples, at effectively equal
data-buffer payload. Padding costs about 48% more, because it materialises
`max(n_times)` for every epoch. On every access and reduction pattern I tried,
including the jagged reduction awkward exists for, the list was substantially
faster.

So I'm not sure what the optional dependency buys. I didn't measure
serialisation, memory-mapping, or interop with a shared representation, which are
the obvious reasons to want it — is that what you had in mind?

One note if it is used: `ak.Array([b.T for b in blocks])` infers
`n_epochs * var * var`, so the channel axis goes variable too. Keeping channels
regular needs `n_epochs * var * n_channels`, which is time-major against our
channel-major convention.

**`warn` + `as_fixed()`.** Right for most methods, I think, with a carve-out for
a few.

`as_fixed()` with `tmin=min, tmax=max` spans the union window, so shorter epochs
get filled. That's padding, and padding is what @agramfort objected to on #12315:
the noise level becomes time-dependent, `nave` becomes a function of time, and
`nave` scales the noise covariance in the inverse. A warning saying a method "may
not behave correctly" doesn't get that across.

I went through all 61 public `BaseEpochs` methods and sorted them by whether
they're mathematically defined by a reduction across a shared time axis. Six are:
`average`, `standard_error`, `iter_evoked`, `subtract_evoked`, `plot_image`,
`plot_topo_image`.
[Full table.](https://github.com/snesmaeili/meta-mne-python-sprint/blob/main/docs/05-method-matrix.md)
The rest are bookkeeping, spatial, per-trial or length-changing, and look cheap
to wrap. So the wrapper could go on broadly, with those six raising or taking an
explicit policy.

### The part the sketch doesn't cover

All four points are about the container and the migration path. The harder half
is what happens once unequal-duration trials have to be compared, and there the
obvious default is wrong in a way that looks right.

Two trials carrying the same 10 Hz oscillation, one 1.0 s and one 2.0 s:

```
native (no warping)          10.0 Hz,  10.0 Hz
warp signal -> TFR           10.0 Hz,  20.0 Hz
TFR -> warp TF axis          10.0 Hz,  10.0 Hz
```

Warping the signal before the transform rescales the frequencies it carries, and
it's the implementation you reach for first. EEGLAB's `newtimef` warps the
representation instead, to the median landmark latencies. Worth deciding on
purpose rather than leaving to follow-ups.

### A cheap starting point

#12311 already lets `Epochs` build events from `raw.annotations.onset`, and logs
that it's ignoring the annotation durations. The durations are sitting in
`Raw.annotations` and get dropped on the way in. Reading them fits the
ndarray-`tmax` direction rather than competing with it.

### Open questions

1. Should scalar `.tmin`/`.tmax` access stay valid in the uniform case?
2. What should reductions do when the contributor count varies over time?
   `Evoked.nave` is scalar. Related: with fixed lengths, weighting ICA and
   covariance by sample and by epoch are the same thing. With variable lengths
   they're different estimators.
3. FIF — pad plus length metadata, or a representation for variable-length
   trials?

#12315 suggested a dev meeting to work through the details. The sprint looks like
a good chance to do that.

@drammock — the assignees from the board card didn't carry over when the draft
was converted to this issue, and I don't have the rights to set them. Could you
add @dnacombo, @virvw, @volerina, @BelizSertcan and @raphbrd back? Tagging them
here in the meantime so they see this.

---

## Before you post

- [ ] read it through and make it yours
- [x] title changed to "Support variable-duration epochs"
- [x] links resolve
- [ ] you can defend the six-method list and the benchmark if asked
- [ ] no claim that the poster validates any of this
- [ ] 38.5% not attributed to Studnicki & Ferris — theirs is 33.3%, 38.5% is your
      cohort's pooled median
- [ ] AI-assistance disclosure goes on the PR description, not here
