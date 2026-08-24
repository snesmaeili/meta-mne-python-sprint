# Problem statement

## The one sentence

> **Raggedness is a data representation; temporal alignment is an analysis decision.**

## Three problems, repeatedly conflated

| | Problem | Issues | What solves it |
|---|---|---|---|
| **P1** | **Container.** Store trials of unequal duration. | #3533, #12315 | A ragged container. This is what the board card names AwkwardArray for. |
| **P2** | **Alignment.** Average processes of unequal duration without averaging misaligned phases. | #5612 | Landmark registration — and, for spectral quantities, in the TF domain. |
| **P3** | **Per-trial time origin.** Rectangular data, different `t=0` per trial. | #5794, #11480 | The degenerate case of P2 where durations are already equal. |

Each previous discussion mixed these up:

- **#3533** argued P1 and was answered with "use rERP", which is a P2 answer.
- **#5612** says explicitly it is *not* about P1 — and gets pointed at #3533.
- **#12315** started as a P1 entry point for PSD and turned into a P1 container
  debate that stalled on a P1 implementation detail.
- **#5794/#11480** (P3) sit unresolved because nobody connected them to P1/P2.

Naming the split up front is the single cheapest thing this proposal does.

## P1 is necessary but not sufficient

NaN-padding — or an awkward array, or a cell array — makes the data
*representable*. It does not make `average()` meaningful. Averaging TFRs across
trials of unequal length averages misaligned processes: time index 350 is not
the same phase of the task in every trial. That is #5612's central point and it
is why a container-only proposal will not satisfy the people who keep asking.

Conversely, warping without a container forces the mne-mobi shape: extract each
trial from `Raw` by hand, loop one-trial `Epochs` objects through the analysis,
and reassemble. That works — it produced a published-quality result — but it
gives up `reject`, `flat`, `reject_by_annotation`, edge handling, `drop_log`,
and every downstream MNE convenience. @larsoner made exactly this point on
#12315: a bottom-up entry point ends up reimplementing `Epochs` without
`Epochs`.

## This is not a gait feature

The gait case is where *we* hit it, and it supplies the cleanest landmarks. The
demand is general:

- **#5612** — mental arithmetic; the problem-solving phase varies per trial and
  per subject. The variable part *is* the process under study.
- **@drammock on #12315** — variable-length spoken sentences, target keyword at
  varying latency, baseline pre-sentence-onset. "Common in language processing,
  and recurrent on the forum."
- **#5794** — tone sequences with variable inter-onset intervals.
- Self-paced and response-terminated tasks generally; reading; decision making;
  sleep segments; rehabilitation; naturalistic and whole-body behaviour.

The same landmark structure recurs everywhere:

```
gait      RHS -> LTO -> LHS -> RTO -> RHS
decision  stimulus -> decision -> response
movement  cue -> movement onset -> movement end
reach     reach onset -> peak velocity -> contact
speech    speech onset -> articulation -> response
cardiac   R -> systole -> diastole
```

## Why the previous attempts stalled

Not for lack of demand, and not for lack of a container idea. Three specific
walls:

1. **The `nave` objection (@agramfort, #12315, 2023-12-20).** Under padding the
   noise level becomes time-dependent, `nave` becomes a function of time, the
   `N=` shown in plots misleads, and `nave` scales the noise covariance used by
   the inverse. This is a statistical objection, not an implementation detail,
   and the archaeology write-ups tend to miss it in favour of the next one.
2. **The SciPy wall (@alexrockhill, #12315).** `np.ma` masked arrays are
   silently stripped by SciPy. Verified still true on SciPy 1.17.1.
3. **The Pandora's-box objection (@kingjr, #3533).** Once the last axis is
   ragged, TFR, PSD, plotting, `times`, covariance, ICA, `nave` and statistics
   all need attention. Correct — if you approach them one at a time.

All three are answerable, and [04-architecture.md](04-architecture.md) answers
them by name. None of them was ever the real blocker, though. @larsoner asked
for a dev meeting on this in 2023 and never got one; the feature has been
short a design, not short an argument.

## What "done" looks like

A user with variable-duration trials can:

1. Epoch them at their true durations, keeping `reject`, `drop_log` and metadata.
2. Filter, baseline, re-reference, run ICA and compute single-trial TFRs without
   thinking about raggedness.
3. Be *stopped* — with a useful message — when they ask for something that needs
   a common time axis.
4. Choose an alignment explicitly, and have the object remember what was chosen.
5. Get a normal `Evoked` / `EpochsTFR` out the far end.

And a reader of the resulting figure can tell, from the object, whether the
frequency axis means what it appears to mean.
