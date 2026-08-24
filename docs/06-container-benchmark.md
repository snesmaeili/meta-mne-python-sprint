# Container benchmark — does AwkwardArray earn a place in MNE?

**Short answer: no.** On EEG-shaped ragged data, `awkward.Array` gives *zero*
memory benefit over a plain `list[np.ndarray]` and is slower on every access
pattern measured — including the vectorised jagged reduction it exists to do.

The board card names AwkwardArray, so this was worth measuring rather than
assuming. Reported as measured. Reproduce with:

```bash
python benchmarks/container_backends.py
```

## Setup

2000 epochs × 128 channels @ 250 Hz, stride durations 0.80–1.70 s (median 1.14) —
one walking session's worth of gait cycles. 573,604 true samples; 850,000 if
padded to the longest epoch, i.e. **32.5% padding waste**.

Environment: Windows 11, Python 3.14.0, NumPy 2.4.1, SciPy 1.17.1, MNE 1.11.0,
awkward 2.13.0.

## Results

All times in ms, best of 3. `get ×500` is 500 random single-epoch accesses.

| backend | payload MB | build | get ×500 | sel epoch | sel chan | iterate | to_dense | reduce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **list** | **587.4** | **2.1** | **0.1** | **0.2** | **98.1** | **2.1** | 299.0 | **99.2** |
| padded | 870.4 | 328.1 | 0.4 | 62.7 | 94.8 | 5.1 | **273.5** | 165.4 |
| awkward | 587.4 | 1226.6 | 67.0 | 0.7 | 616.2 | 318.4 | 672.0 | 1192.7 |

Relative to `list`:

| | memory | random access | to_dense | reduce |
|---|---:|---:|---:|---:|
| padded | 1.48× | 5.0× slower | 0.91× | 1.7× slower |
| awkward | **1.00×** | **759× slower** | 2.3× slower | **12× slower** |

`reduce` is the per-epoch mean over time — a reduction across the jagged
structure, which is precisely what awkward is built for. All three
implementations agree to 1.1e-16, so this is a like-for-like comparison. Awkward
loses it by 12×.

## Why awkward loses

**No memory to save.** Awkward's payload is byte-identical to the list's
(587.4 MB both). Both store exactly the true samples and nothing else. Awkward
saves memory relative to *padding*, but so does a list — and the list gets there
without a new dependency.

**One ragged axis, not many.** Awkward is designed for deeply nested,
heterogeneous, variable-depth structures. EEG epochs are ragged along exactly
one axis, with a fixed channel count and a fixed dtype. That is the easiest
possible case, and it is comfortably handled by an array per epoch.

**Layout fights MNE's convention.** The only awkward type that structurally
enforces the invariant is:

```
2000 * var * 128 * float64        # (epoch, time, channel)
```

Putting `var` at the epoch level and keeping channels as a `RegularArray` is
what guarantees that *only time is ragged*. But that is time-major, while MNE
is channel-major `(n_channels, n_times)` everywhere. Every single-epoch access
pays a transpose, which is where the 759× comes from.

**Modest epoch counts.** At 2000 epochs, Python loop overhead is ~2000
iterations of negligible cost, while each per-block NumPy call operates on
contiguous memory. Awkward's vectorisation would need orders of magnitude more
list elements to amortise its indirection.

## The layout trap — worth flagging to maintainers

The intuitive construction is wrong, silently:

```python
ak.Array([b.T for b in blocks])     # -> 2000 * var * var * float64
```

Both dimensions become `var`. Nothing then prevents channel 0 and channel 1 of
the same epoch from having different lengths — the invariant the container
exists to enforce is gone, with no error. It is also ~1000× slower to build and
~1000× slower to index, because every epoch becomes a separate Python-level
list instead of a view into one buffer.

The correct construction goes through a flat buffer:

```python
flat = np.concatenate([b.T for b in blocks], axis=0)   # (total_samples, n_ch)
array = ak.unflatten(ak.from_numpy(flat, regulararray=True), counts, axis=0)
# -> 2000 * var * 128 * float64
```

Anyone proposing awkward for MNE needs to know this. Both variants are
implemented and the wrong one is documented in `ragged_epochs/_backends.py`.

## Recommendation

**Use `list[np.ndarray]` plus an offsets array.** Zero new dependencies,
channel-major so it matches MNE everywhere, fastest on every access pattern,
and identical memory to the most sophisticated alternative.

Keep **padded + explicit lengths** as the interchange and IO format — it is what
FIF would need and what mne-connectivity #142 already shipped for its ragged
case. Note the validity information must live in a separate `lengths` array,
never in a `np.ma.MaskedArray`: SciPy silently strips masks (verified on 1.17.1,
`scipy.signal.spectrogram` returns a plain `ndarray` and treats padded samples
as real data). That is the blocker alexrockhill hit on #12315 and it is still
live.

**This does not close the door on awkward.** If a future use case has genuinely
nested raggedness — ragged channels *and* ragged time, as in the
channel-specific epoch removal of #11705/#12219 — the calculus changes. The
finding is scoped: for one ragged axis at EEG scale, awkward costs and does not
pay.

## Caveats

- One machine, one data shape. The conclusion is robust on the memory axis
  (identical payload is a structural fact, not a timing artefact); the timing
  ratios will vary.
- Serialisation and memory-mapping are not measured. Awkward's single
  contiguous buffer may do better there, which matters for the IO layer.
- Compression is not measured. For on-disk formats, padding waste largely
  disappears under compression, which weakens the memory argument for *any*
  ragged container in the IO path specifically.
