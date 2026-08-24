# Prior art

What already exists, in other toolboxes and in the literature. Maintainers will
raise most of this; better to have engaged with it first.

## EEGLAB — no ragged container, alignment at the analysis layer

EEGLAB has fixed-length epochs and solves raggedness entirely in `newtimef`'s
`timewarp` option. The mechanics, from `timefreq.m`:

```matlab
timemarks = g.timestretch{1}';
timerefs  = median(g.timestretch{1}', 2);   % <- default target
M   = timewarp(marksPos, refsPos);          % linear warp matrix
r   = sqrt(mytmpall .* conj(mytmpall));     % magnitude
TSr = transpose(M * r');                    % warp magnitude
TStheta = angtimewarp(...);                 % warp phase CIRCULARLY
TStmpall = TSr .* exp(i * TStheta);         % recombine
```

Three things to take from this:

1. **Warping happens after the time-frequency transform**, described in the
   source as "following time/freq transform but before smoothing across trials".
2. **The default target is the median** of each event's latencies across trials.
   With two datasets, the grand median of both.
3. **Phase is warped circularly, magnitude linearly.** Linear interpolation
   between +179° and −179° gives 0°, not 180°, which is why `angtimewarp` exists
   separately.

This is the reference implementation, and it is what the gait-EEG literature
means by the standard phrasing: ERSPs "linearly time-warped using EEGLAB's
`timewarp()` function to the group median gait cycle length using foot lift-off
and contact events."

## FieldTrip — ragged container, no alignment layer

`data.trial` and `data.time` are **cell arrays** of variable-length matrices.
Raggedness is solved by not using a rectangular array at all — the oldest and
simplest answer. `ft_freqanalysis` explicitly allocates for a variable number of
tapers per trial. Downstream, the documented path is NaN-padding plus
`cfg.nanmean = 'yes'` in `ft_selectdata`.

@mmagnuski's comment on #3533 — that FieldTrip's variable-length trials "were
never that useful in practice" — deserves engaging rather than dismissing. The
reading that fits the evidence: FieldTrip built layer 1 and stopped. Without an
alignment layer, users hit the same wall one level later, at the point where
they want to average. That is an argument for layers 2–3, not against layer 1.

## mne-connectivity #142 — the ecosystem's one shipped ragged decision

Ragged multivariate connectivity seeds/targets, solved by **padding**. The only
place the MNE ecosystem has actually made and shipped a ragged-data call.

Correct for that problem: there is no time axis, so padding carries no
alignment semantics. Cite it as precedent for "padding is fine *at the
computation boundary*", which is exactly this design's position — not as
precedent for padding as the container.

## `unfold` — the counter-proposal

Ehinger & Dimigen (2019), *PeerJ* 7:e7838. Linear deconvolution plus spline
regression (generalised additive modelling) for temporally overlapping
responses and continuous covariates. Time expansion via stick functions or
time-splines; the resulting "regression-ERPs" are analysed like ordinary ERPs.

This is the modern form of @kingjr's "just use rERP" on #3533, and it will be
the first comment on any new proposal. The honest position:

- Deconvolution solves **overlap** and **continuous covariates**. If trial
  duration is a nuisance variable you want to regress out, it is the better
  tool.
- It does not give you the **ERSP of a variable-duration process aligned to its
  own internal landmarks**. There is no rERP formulation that puts toe-off at
  12% of the gait cycle.
- They are complementary. Say which question goes to which tool. Do not claim
  superiority.

## Landmark registration — the formal name

What EEGLAB calls time warping is **landmark registration** in functional data
analysis, with a statistics literature going back three decades:

- Kneip & Gasser (1992) — landmark registration
- Ramsay & Li (1998) — Procrustes / continuous registration
- Ramsay & Silverman (2005) — *Functional Data Analysis*, the standard reference

The framing: prominent features (peaks, valleys, events) do not occur at the
same times across curves. This is **phase variation**, as distinct from
amplitude variation. A cross-sectional mean computed without accounting for
phase variation distorts exactly the features you care about — which is the
formal statement of #5612's objection to NaN-padding.

Useful consequences:

- **`scikit-fda`** (BSD) implements landmark registration in Python. Evaluate as
  a dependency rather than reimplementing. It also brings continuous
  registration, which is the principled version of "warp without landmarks".
- The warping function must be **strictly increasing and smooth**. Our
  `piecewise_linear_warp` enforces strict monotonicity and refuses otherwise.
- Continuous alternatives: DTW (Sakoe & Chiba 1978; Wang & Gasser 1997),
  Fisher-Rao / SRVF (Srivastava et al. 2011).

## Gait-EEG literature

- **Gwin, Gramann, Makeig & Ferris (2011)**, *NeuroImage* 54:1289–1296. The
  canonical paper: spectrograms time-locked to the gait cycle; electrocortical
  sources in anterior cingulate, posterior parietal and sensorimotor cortex show
  significant intra-stride spectral changes.
- **Studnicki & Ferris (2023)**, *eNeuro* 10(4) ENEURO.0463-22.2023.
  Parieto-occipital dynamics during real-world table tennis — a variable-length
  "rally cycle" with ball contact as an internal landmark. This is the result
  our poster reproduced.
- Standard practice across the field: linear time warping to the group median
  cycle length using foot lift-off and contact events.

**The weakness to disclose.** Averaging time-warped ERSPs across subjects with
differing gait characteristics can shift or reverse the phase of power changes
relative to single-subject data. Warping is not free, and a proposal that
presents it as free will be correctly distrusted.

## RIDE — adjacent, not the same problem

Residue iteration decomposition handles trial-to-trial ERP *component latency*
variability by decomposing and realigning latent components. Related in spirit,
different in kind: it is about latent structure within a fixed epoch, not about
epochs of different duration. Background, not architecture.

## Awkward Array / `ragged`

`awkward` (Scikit-HEP) is built for nested, variable-length, heterogeneous data
— its home domain is high-energy physics event records. `scikit-hep/ragged`
wraps it in an Array-API-compliant interface, which matters because
SciPy and scikit-learn are moving toward that standard.

Awkward's own documentation is clear that algorithms requiring rectangular input
still need an explicit strategy (crop, reduce, or pad). That is the same
boundary this design draws.

For the measured verdict on whether it earns a place in MNE, see
[06-container-benchmark.md](06-container-benchmark.md). Short version: for one
ragged axis at EEG scale, it costs and does not pay.

## Nothing in mne-tools has ever mentioned AwkwardArray

`org:mne-tools awkwardarray` → 0 hits. `org:mne-tools "awkward array"` → 1 hit,
a false positive (mne-connectivity #142, about ragged seeds/targets, not the
library). The ~47 mne-python hits for `awkward` are the ordinary English word.

The library proposal is new. The underlying feature request is not.
