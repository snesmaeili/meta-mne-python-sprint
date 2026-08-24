# Roadmap

**Only stage 1 is sprint scope.** The rest exists so the design can be shown to
close, and so reviewers can see where each piece lands. Presenting six stages as
"the plan" invites the correct response that it is too big to start.

Stages 2–5 are already implemented in the prototype (`ragged_epochs/`) as
evidence. That is not the same as being ready to land in core, and the proposal
should not pretend otherwise.

---

## Stage 1 — representation *(sprint scope)*

Per-epoch bounds, ragged storage, indexing / drop / metadata / channel
operations, `durations`, per-epoch times, explicit `get_data(representation=)`,
and the backend benchmark. **No warping at all.**

- Entry point: honour `Annotations.duration` in `Epochs` construction. Already
  in the data model, already documented as ignored, already the thing #12315 was
  blocked on. Smallest reviewable change available.
- `times` raises on ragged data. The single highest-risk API decision; settle it
  before anything is built on top.
- Backend: `list[np.ndarray]` + offsets, per
  [06-container-benchmark.md](06-container-benchmark.md).
- Deliverable already in hand: the method matrix, so reviewers can see the total
  blast radius (61 methods, 48 of them unaffected) rather than guessing at it.

Open questions for the sprint, not decidable unilaterally:

- Public class name: extend `Epochs`, or a separate `RaggedEpochs`? The
  prototype uses a separate class to avoid silently changing a return type, but
  this is a maintainer call.
- Does `Epochs` with equal durations *become* the ragged class, or stay
  separate? The prototype asserts bit-identical behaviour in the uniform case,
  which makes either choice defensible.
- FIF: pad + a lengths tag, or a new block type?

## Stage 2 — ragged-native signal operations

Baseline, projection, referencing, filter, detrend, Hilbert, decimation,
resampling, rejection, interpolation. Each validated against stock MNE run
per-epoch — that parity oracle is what keeps "no kernel rewrites" honest.

Watch for: filter edge effects and Hilbert padding are duration-dependent. True
of fixed-length epoching too, just decided once at `tmin`/`tmax` time instead of
per trial. Document rather than hide.

## Stage 3 — ragged-native analyses with declared policy

PSD, ICA, covariance, source estimation. The work here is **semantic, not
numerical**: sample- vs epoch-weighting, and the common frequency grid.

This is the stage with the most scientific value and the least code. Fixed-length
epochs silently enforce `sample weighting == epoch weighting`; ragged trials
break the identity. Getting the API to state the choice — rather than pick one —
is the deliverable.

## Stage 4 — ragged TFR

`compute_tfr(average=False)` returning a ragged-time representation with a
common frequency axis. No cross-trial average unless times are compatible.

Includes the policy check the prototype already has: the **shortest** epoch
bounds which frequencies are computable at all, because every epoch must yield
the same frequency axis. With `n_cycles ∝ freq` the wavelet length is
frequency-independent, so raising `fmin` does not help — the error has to say
so.

## Stage 5 — alignment API

Common crop, explicit padding (returning `nave(t)`), duration normalisation,
landmark piecewise registration, and **TFR-domain landmark warping** following
EEGLAB/Gwin.

Evaluate `scikit-fda` (BSD) rather than reimplementing landmark registration.

DTW and other signal-driven registration stay out. They belong as pluggable
algorithms once the landmark path is established, not in the first design.

## Stage 6 — common-grid ecosystem and IO

`Evoked`, ITC, plots, decoding, statistics, `to_data_frame`, FIF serialisation
and export. Mostly a matter of routing: these are the six methods that require a
common time axis, plus the two IO methods.

`plot` deserves attention — see #10367 on epoch time-axis labelling, which is
the visualisation face of the same problem.

---

## Sequencing note

Stages 2 and 3 are independent of each other and both depend only on stage 1.
Stage 5 depends on stage 1; stage 4 depends on 1 and benefits from 5. Nothing
after stage 1 is on the critical path for the sprint.

Several of these are themselves multiple PRs. A single mega-PR would be
unreviewable and is not proposed.
