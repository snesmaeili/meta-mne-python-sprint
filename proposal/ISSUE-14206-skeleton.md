# Writing guide for #14206 — after larsoner's design comment

**Write the sentences yourself.** MNE's policy is that AI-generated prose should
not be pasted into issue or PR descriptions.

Issue: https://github.com/mne-tools/mne-python/issues/14206
State: empty body, no assignees, one comment from @larsoner (2026-08-24).

---

## What changed

@larsoner posted a design sketch:

1. `tmin` and `tmax` allowed to be ndarray of shape `(n_events,)`
2. FIF I/O — "cautiously optimistic it won't be too hard"
3. `awkward` as an **optional** dep, `_soft_import`ed when `tmin`/`tmax` are ndarray
4. Expects breakage; wrap methods with a warning plus `self.as_fixed().<meth>`,
   where `as_fixed()` returns an `EpochsArray` with `tmin=np.min(self.tmin),
   tmax=np.max(self.tmax)`; follow-up PRs improve methods gradually

**Note what it does and does not cover.** All four points are about the
*container* and the *migration path*. It says nothing about alignment: no
landmark warping, no warp target, no signal-versus-TFR domain, no reduction
semantics beyond "expect breakage".

So the two designs overlap on one axis and are complementary on the rest. Write
the issue that way: concede the overlap, then contribute what he did not cover
and the evidence he does not have.

---

## Concede this one cleanly

**The container should be one class with ndarray `tmin`/`tmax`, not a separate
`RaggedEpochs`.** Say so plainly and early. Reasons worth stating, because they
show you evaluated it rather than just agreeing:

- a separate class means duplicating or awkwardly inheriting 61 public methods
- the objection that array `tmax` silently changes what the constructor returns
  does not survive contact — you only get arrays if you pass arrays, so it is
  opt-in at construction, not silent

Conceding fast costs nothing and buys credibility for the three positions below.
The prototype's separate class becomes a design probe, not a proposal.

One narrow question survives: once `epochs.tmin` can be an array, code that
receives an `Epochs` from elsewhere and does scalar arithmetic on `.tmin` /
`.tmax` changes behaviour. Ask whether scalar access should stay valid in the
uniform case. Migration-surface question, not an objection.

---

## Three positions to state, not ask

### P1 — On `awkward`, there is a measurement

If it is `_soft_import`ed, a working fallback has to exist anyway. Measured at
EEG scale (2000 epochs × 128 ch @ 250 Hz, durations 0.8–1.7 s), a plain list of
arrays plus offsets stored only the observed samples — the same data-buffer
payload as Awkward — and was substantially faster on every access and reduction
pattern tested, including the jagged reduction Awkward exists for.

State the implication: on this evidence the optional dependency may not be needed
at all. Then name what you did **not** measure, because that is likely where his
reasons live — serialization, memory-mapping, interop with a shared ecosystem
representation.

Wording discipline:

- "data-buffer payloads were effectively equal at this scale", not "byte
  identical" — the figure excludes Python object overhead and Awkward's
  offset/layout buffers
- describe it as one synthetic EEG-scale microbenchmark, environment in the
  linked repo
- the layout point as a schema complication, not a trap:
  `ak.Array([b.T for b in blocks])` gives `n_epochs * var * var`, so the channel
  axis also becomes variable; enforcing "only time is ragged" needs
  `n_epochs * var * n_channels`, which is time-major against MNE's
  channel-major convention

### P2 — `as_fixed()` needs a carve-out, and you can name which methods

The strongest technical point, and it *supports* his incremental strategy rather
than opposing it.

`as_fixed()` with `tmin=min, tmax=max` spans the union window, so shorter epochs
get filled. That is padding, and padding is what @agramfort objected to on
#12315: the noise level becomes time-dependent, `nave` becomes a function of
time, the `N=` shown in plots misleads, and `nave` scales the noise covariance
used by the inverse. A warning that a method "may not behave correctly" does not
convey "your effective N varies across the epoch".

The proposal: `warn + as_fixed()` is right for bookkeeping, spatial and plotting
methods, and needs different handling for reductions. Only a small subset of the
public API is mathematically defined by a reduction across a shared time axis —
`average`, `standard_error`, `iter_evoked`, `subtract_evoked`, `plot_image`,
`plot_topo_image`. Link the generated matrix; it is exactly the list his point 4
needs in order to be applied selectively.

Offer the prototype's policy as one option rather than the answer: `pad()`
returns the time-resolved contributor count alongside the data, so the variation
stays visible. Ask whether that is the right abstraction given `Evoked.nave` is
scalar.

### P3 — Alignment is the layer the sketch leaves out

Say directly that his points address representation and migration, and that the
harder half is what happens once trials of different duration have to be
compared. Then one result that makes it concrete.

Two trials, same 10 Hz oscillation, 1.0 s and 2.0 s:

```
native (no warping)          10.0 Hz,  10.0 Hz
warp signal -> TFR           10.0 Hz,  20.0 Hz
TFR -> warp TF axis          10.0 Hz,  10.0 Hz
```

Two sentences after it, no more. Warping the signal before the transform
rescales the frequencies of the oscillations it carries, and it is the
implementation someone reaches for first; EEGLAB's `newtimef` warps the
representation instead, to the median landmark latencies. The point: alignment
has a wrong default that looks correct, so it needs design attention rather than
being deferred to follow-up PRs.

This is why the three-problem separation matters — representation (#3533,
#12315), within-trial temporal anchors (#5794 needs a different zero point per
trial, #11480 needs a baseline defined by a *different* event in the same
trial), and alignment/reduction (#5612). Do not call the middle one "per-trial
time origin"; that only covers #5794. Do not say every previous thread merged
them — #5612 tried hard to keep them separate. Safer: the discussions repeatedly
cross between the three even when the original request concerns only one.

---

## Suggested shape (~700–900 words)

1. **Opening (~60 words).** What the issue consolidates. Thank Eric for the
   sketch, say you are responding to it, note there is an executable prototype
   used to probe the design space — explicitly not a merge proposal.
2. **Agreement on the container (~80 words).** Concede single-class,
   ndarray-`tmin`/`tmax`, and why. Raise the scalar-access question.
3. **The three problems (~90 words).** Sets up section 5.
4. **Use cases (~110 words).** Mental arithmetic (#5612, open since 2018 — the
   variable period *is* the object of study). Spoken sentences (@drammock on
   #12315, raised against the "rare use case" argument). Tone sequences (#5794).
   The 2022 independent rediscovery in #5612, where a gait user and the
   arithmetic-task author converge on long fixed epoch → `EpochsTFR` → crop →
   interpolate. Your gait work in **2–3 sentences maximum** — same architectural
   problem, not a request for a gait feature.
5. **P1, P2, P3 (~350 words).** The core.
6. **Starting point (~50 words).** #12311 already builds events from
   `raw.annotations.onset` and explicitly logs that annotation durations are
   ignored. Precise wording: the durations are already in `Raw.annotations` and
   `Epochs` discards them. Cheap first step that fits his direction.
7. **Links (tiny).** Sprint repo, `ragged-epochs` branch, draft PR to follow.

**Closing.** #12315 called for a dev meeting on the details; the sprint is a good
opportunity to have it. Do not claim he never got one.

---

## Also do

- Retitle to **`Support variable-duration epochs`**. Awkward is explicitly an
  optional implementation detail in his sketch, so the headline overstates it.
- Restore assignees: dnacombo, virvw, volerina, BelizSertcan, raphbrd.

## Do not claim

- that the poster validates `mne.ragged` — it came from `eneuro_merged_ersp.py`,
  and the reproduction script has not been run
- that 38.5% is Studnicki & Ferris's value — it is your cohort's pooled median;
  theirs is 33.3%
- that the per-trial operations are fast — they are a Python loop
- that Awkward "does not work" — it works, and lost on measurement at this scale
  in the patterns tested
- SciPy: reproduced on **1.17.1 and 1.18.1** (1.18.1 released 2026-08-21).
  Do not say "still true" on the basis of 1.17.1 alone.
