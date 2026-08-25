Took the type-hint suggestion, thanks.

I spiked `EpochsRagged(BaseEpochs)` to check. It does not avoid the shared-module
changes: with `mixin.py` and `channels.py` back at upstream, `epochs[0:2]` dies in
`_getitem` (`copyto` on a list) and `pick` dies in `_pick_drop_channels`
(`list.take`). Removing those means overriding both, so 64 and 48 lines of
selection/drop_log/metadata bookkeeping copied into the subclass, which would
drift. `shift_time` is the one clean win: the ragged path is a whole early return,
so an override is 20 lines. Today it is 5 branches in shared code and 22 in
`epochs.py`, and most of the 22 are construction and per-epoch bounds that would
move to the subclass rather than disappear. So I lean to one class, but I will
switch if you would rather have the duplication than the branches.

Dead end worth recording: an object-dtype `_data` makes `_getitem` work untouched,
but `np.copyto` then copies references, so `epochs[0:2]` comes back sharing memory
with the parent. Kept the list.

The decorator mechanics are separable from that either way. I can swap the
`setattr` loops for an explicit call at the top of each affected method, so it
reads where you would look for it. Happy to do that first.

Tutorial: `sleep_physionet` fits, and no new dataset is needed. Its hypnogram
annotations already carry durations, and `tutorials/clinical/60_sleep.py` passes
`chunk_duration=30.` to cut them into fixed epochs. On SC4001 that turns 141 sleep
stage bouts spanning 30 to 1890 s into 653 identical 30 s chunks, so the workaround
this removes is already sitting in our own docs. I looked at openneuro-py first,
but the dataset I validated against publishes `duration = n/a` and its cycles come
from an IMU rule that does not belong in a tutorial.
