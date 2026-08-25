# Draft replies to the inline threads on #14210

Not posted. No AI footer — the bot asked for a "Addressed by Claude Code" line,
which is exactly the trace we keep out.

---

## Thread 3850822427 — "Not sure why this would need to be string?"

Done, unquoted in both signatures. 3.11 is the floor so PEP 604 resolves at
runtime, and the parameter two lines up already wrote `data: np.ndarray | None`
without quotes.

---

## Thread 3851697997 — decorators are fine; `verbose="error"` for the example

Thanks. Added `verbose="error"` to the EDF read, same as `60_sleep.py` does for
these files.

The build was failing on a second thing as well: numpydoc emits a bare `py:obj`
reference for each public property, and `BaseEpochs` uses the no-members
template, so those references rely on the `nitpick_ignore_regex` that already
lists `times`, `tmin`, `filename`. `durations` and `variable_duration` were
missing from it. Matching is `fullmatch`, so the `duration` entry did not cover
`durations`.

`ty` was red too, since `epochs.py` is in `[tool.ty.src]`. Two causes: the
per-epoch bounds are `None` on the fixed path, and `_data` holds a list when
durations vary. Both attributes are declared on the class now, with three
suppressions where the code deliberately departs from that. The signatures also
say what they already accepted, a per-epoch list for `data` and one value per
event for `tmin`.

Two real defects came out of that pass. `apply_function` reached
`self._data.dtype` and then indexed the list with a tuple, so it raised
`TypeError` from inside rather than declining; it is in the not-implemented
table now. And `_pick_drop_channels` assigned a list to `self._data` on
`UpdateChannelsMixin`, which made that attribute a list for `Evoked` too and
produced ten errors in `evoked.py`; it replaces the list contents instead.

---

## Thread 3851703389 — killed the running CIs

Pushed. `Autofix and style` and `build_docs` are both green.

---

## Thread 3850843037 — EpochsRagged vs one class

No reply needed; you closed it with "sounds good to me" and the decorators stay.
