"""The data matrix: the axes @drammock named, plus the ones the code invites.

Kept in one place so the matrix sweep and the fuzzer draw from the same pool and
a finding always names a spec that can be rebuilt.
"""

from .build import Spec

# NB: names are about how a spec is *built*, not which code path it takes.
# ``_check_variable_bounds`` collapses bounds that carry no actual variation
# back to the scalar path, so ``equal_ragged_path`` and ``all_one_sample`` both
# report ``variable_duration == False`` despite being built from a list. Check
# ``build(spec).epochs.variable_duration`` before calling a result ragged-only.

# -- duration spread --------------------------------------------------------
SPREAD = [
    Spec(lengths=(100, 100, 100, 100), name="equal_ragged_path"),
    Spec(lengths=(100, 100, 100, 100), force_fixed=True, name="equal_fixed_path"),
    Spec(lengths=(100, 101, 100, 99), name="off_by_one"),
    Spec(lengths=(100, 250, 75, 180), name="reference_fixture"),
    Spec(lengths=(100, 200, 100, 200), name="two_to_one"),
    Spec(lengths=(3, 300, 3, 300), name="hundred_to_one"),
    Spec(lengths=(1, 100, 100, 100), name="one_sample_first"),
    Spec(lengths=(100, 100, 100, 1), name="one_sample_last"),
    Spec(lengths=(2, 3, 2, 3), name="all_tiny"),
    Spec(lengths=(3000, 10, 10, 10), name="longest_first"),
    Spec(lengths=(10, 10, 10, 3000), name="longest_last"),
]

# -- how many epochs --------------------------------------------------------
COUNTS = [
    Spec(lengths=(137,), name="single_epoch"),
    Spec(lengths=(137, 42), name="two_epochs"),
    Spec(lengths=(137, 42, 211), name="three_epochs"),
    Spec(lengths=tuple(50 + (i * 37) % 200 for i in range(50)), name="fifty_epochs"),
    Spec(lengths=tuple(20 + (i * 13) % 90 for i in range(500)), name="five_hundred"),
    Spec(lengths=tuple(10 + (i * 7) % 40 for i in range(2000)), name="two_thousand"),
]

# -- channels ---------------------------------------------------------------
CHANNELS = [
    Spec(lengths=(100, 250, 75), n_channels=1, name="one_channel"),
    Spec(lengths=(100, 250, 75), n_channels=2, name="two_channels"),
    Spec(lengths=(100, 250, 75), n_channels=64, name="sixty_four_channels"),
    Spec(lengths=(100, 250, 75), n_channels=306, name="three_o_six_channels"),
    Spec(
        lengths=(100, 250, 75),
        n_channels=16,
        mixed_types=True,
        name="mixed_ch_types",
    ),
    Spec(
        lengths=(100, 250, 75),
        n_channels=16,
        mixed_types=True,
        n_bads=3,
        name="mixed_with_bads",
    ),
]

# -- tmin ------------------------------------------------------------------
TMINS = [
    Spec(lengths=(100, 250, 75, 180), tmin="zero", name="tmin_zero"),
    Spec(lengths=(100, 250, 75, 180), tmin="negative", name="tmin_negative"),
    Spec(lengths=(100, 250, 75, 180), tmin="positive", name="tmin_positive"),
    Spec(lengths=(100, 250, 75, 180), tmin="mixed", name="tmin_mixed"),
]

# -- dropped ----------------------------------------------------------------
DROPPED = [
    Spec(lengths=(100, 250, 75, 180, 120), drop=(1,), name="drop_middle"),
    Spec(lengths=(100, 250, 75, 180, 120), drop=(0,), name="drop_first"),
    Spec(lengths=(100, 250, 75, 180, 120), drop=(4,), name="drop_last"),
    Spec(lengths=(100, 250, 75, 180, 120), drop=(1, 3), name="drop_noncontiguous"),
    Spec(lengths=(100, 250, 75, 180, 120), drop=(0, 1, 2), name="drop_down_to_two"),
    Spec(lengths=(100, 250, 75, 180, 120), drop=(0, 1, 2, 3), name="drop_down_to_one"),
]

# -- sampling rate ----------------------------------------------------------
SFREQS = [
    Spec(lengths=(100, 250, 75, 180), sfreq=10.0, name="sfreq_10"),
    Spec(lengths=(100, 250, 75, 180), sfreq=100.0, name="sfreq_100"),
    Spec(lengths=(100, 250, 75, 180), sfreq=250.0, name="sfreq_250"),
    Spec(lengths=(100, 250, 75, 180), sfreq=512.3, name="sfreq_noninteger"),
    Spec(lengths=(100, 250, 75, 180), sfreq=1000.0, name="sfreq_1000"),
    # the two sfreq corners the ``sampling_period`` nudge has to survive:
    # a 1-sample-wide epoch at the lowest rate, and a rate whose boundary
    # times are not exactly representable
    Spec(lengths=(1, 4, 1, 4), sfreq=10.0, name="sfreq_10_tiny"),
    Spec(lengths=(2, 3, 2, 3), sfreq=10.0, name="sfreq_10_all_tiny"),
    Spec(lengths=(3, 300, 3, 300), sfreq=512.3, name="sfreq_noninteger_spread"),
    Spec(lengths=(2, 3, 2, 3), sfreq=512.3, name="sfreq_noninteger_tiny"),
    Spec(
        lengths=tuple(10 + (i * 7) % 40 for i in range(2000)),
        sfreq=512.3,
        name="two_thousand_noninteger",
    ),
]

# -- window arithmetic corners ---------------------------------------------
#
# ``n_epochs`` is clamped to ``len(epochs)`` at plot time, so the interesting
# combinations are driven from ``PLOT_KWARGS`` below; these are the data-side
# corners the window arithmetic has to survive.
WINDOWS = [
    Spec(lengths=(137,), name="single_epoch_w"),
    Spec(lengths=(1, 137), name="one_sample_then_long"),
    Spec(lengths=(137, 1), name="long_then_one_sample"),
    Spec(lengths=(100, 1, 100, 100), name="one_sample_middle"),
    Spec(lengths=(1, 1, 1, 1), name="all_one_sample"),
    Spec(lengths=(1, 1, 1, 1), force_fixed=True, name="all_one_sample_fixed"),
    Spec(lengths=(2, 2, 2, 2), force_fixed=True, name="all_two_sample_fixed"),
]

ALL = SPREAD + COUNTS + CHANNELS + TMINS + DROPPED + SFREQS + WINDOWS

#: Cheap enough to run every one of them under every action script.
FAST = [
    s
    for s in ALL
    if len(s.lengths) <= 50 and s.n_channels <= 64 and max(s.lengths) <= 500
]

BY_NAME = {s.label(): s for s in ALL}


# -- plot() argument variations ---------------------------------------------
PLOT_KWARGS = {
    "default": dict(n_epochs=2),
    "one_epoch_window": dict(n_epochs=1),
    "window_bigger_than_data": dict(n_epochs=99),
    "events": dict(n_epochs=2, events=True),
    "butterfly": dict(n_epochs=2, butterfly=True),
    "decim_2": dict(n_epochs=2, decim=2),
    "decim_4": dict(n_epochs=2, decim=4),
    "scalings_dict": dict(n_epochs=2, scalings=dict(eeg=20e-6)),
    "group_by_selection": dict(n_epochs=2, group_by="selection"),
    "group_by_position": dict(n_epochs=2, group_by="position"),
}
