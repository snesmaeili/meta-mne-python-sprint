"""V5 -- settle the container question by measurement.

The board card names AwkwardArray. This benchmark exists so that choice is
made on numbers rather than on the fact that awkward is the library people
reach for when they hear "ragged".

Measured at realistic MoBI scale: gait cycles from a walking session, 0.8-2.0 s
at 250 Hz, 128 channels. Reported either way -- "we benchmarked and a plain
list wins" is a legitimate and useful sprint outcome, and a much better answer
to give maintainers than "we assume awkward".

Run:  python benchmarks/container_backends.py
"""

from __future__ import annotations

import gc
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from ragged_epochs._backends import BACKENDS  # noqa: E402

SFREQ = 250.0
N_CHANNELS = 128
N_EPOCHS = 2000
DUR_LO, DUR_HI = 0.8, 2.0
N_REPEAT = 3


def make_blocks(n_epochs=N_EPOCHS, n_channels=N_CHANNELS, seed=0):
    """Gait-like ragged data: log-normal-ish stride durations in [0.8, 2.0] s."""
    rng = np.random.default_rng(seed)
    durations = np.clip(rng.normal(1.15, 0.18, n_epochs), DUR_LO, DUR_HI)
    return [
        rng.standard_normal((n_channels, int(round(d * SFREQ)))) * 1e-6
        for d in durations
    ], durations


def timeit(fn, n=N_REPEAT):
    """Best-of-n wall time in milliseconds."""
    best = float("inf")
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def bench_backend(name, blocks):
    cls = BACKENDS[name]
    row = {"backend": name}

    t0 = time.perf_counter()
    store = cls.from_list(blocks)
    row["build_ms"] = (time.perf_counter() - t0) * 1e3
    row["payload_mb"] = store.nbytes / 1e6

    n = len(store)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, n, 500)
    sel = np.sort(rng.choice(n, n // 4, replace=False))
    chans = np.arange(0, N_CHANNELS, 4)

    row["get_500_ms"] = timeit(lambda: [store.get(int(i)) for i in idx])
    row["select_epochs_ms"] = timeit(lambda: store.select_epochs(sel))
    row["select_channels_ms"] = timeit(lambda: store.select_channels(chans))
    row["iterate_all_ms"] = timeit(lambda: [b.shape for b in store], n=1)
    row["to_dense_ms"] = timeit(lambda: store.to_dense(), n=1)
    row["dense_mb"] = (n * store.n_channels * int(store.lengths.max()) * 8) / 1e6

    # The operation awkward is actually built for: a reduction across the
    # jagged structure without a Python-level loop. If awkward is going to win
    # anywhere, it is here, so give it a fair shot rather than only measuring
    # the access patterns that suit a list of arrays.
    row["reduce_ms"] = timeit(lambda: _per_epoch_mean(store), n=1)
    if hasattr(store, "typestr"):
        row["typestr"] = store.typestr
    return row


def _per_epoch_mean(store):
    """Mean over time for every epoch -> (n_epochs, n_channels)."""
    if store.name == "awkward":
        import awkward as ak

        return ak.to_numpy(ak.mean(store._array, axis=1))
    if store.name == "padded":
        out = np.empty((len(store), store.n_channels))
        for i, k in enumerate(store.lengths):
            out[i] = store._data[i, :, :k].mean(axis=1)
        return out
    return np.stack([b.mean(axis=1) for b in store])


def main():
    blocks, durations = make_blocks()
    total_samples = sum(b.shape[1] for b in blocks)
    n_max = max(b.shape[1] for b in blocks)

    print("V5 -- container backend benchmark")
    print("=" * 74)
    print(
        f"{N_EPOCHS} epochs x {N_CHANNELS} channels @ {SFREQ:g} Hz, "
        f"durations {durations.min():.2f}-{durations.max():.2f} s "
        f"(median {np.median(durations):.2f})"
    )
    print(
        f"true samples: {total_samples:,}   "
        f"padded to max: {N_EPOCHS * n_max:,}   "
        f"waste: {100 * (N_EPOCHS * n_max - total_samples) / (N_EPOCHS * n_max):.1f}%"
    )
    print()

    rows = []
    for name in ("list", "padded", "awkward"):
        try:
            rows.append(bench_backend(name, blocks))
        except ImportError as exc:
            print(f"  skipping {name}: {exc}")

    cols = [
        ("backend", "backend", "{:<8}"),
        ("payload_mb", "payload MB", "{:>10.1f}"),
        ("build_ms", "build ms", "{:>9.0f}"),
        ("get_500_ms", "get x500", "{:>9.1f}"),
        ("select_epochs_ms", "sel epoch", "{:>10.1f}"),
        ("select_channels_ms", "sel chan", "{:>9.1f}"),
        ("iterate_all_ms", "iterate", "{:>8.0f}"),
        ("to_dense_ms", "to_dense", "{:>9.0f}"),
        ("reduce_ms", "reduce", "{:>9.1f}"),
    ]
    W = 12
    print("".join(h.rjust(W) for _, h, _ in cols))
    print("-" * (W * len(cols)))
    for r in rows:
        print("".join(
            (f"{r[k]:,.1f}" if isinstance(r[k], float) else str(r[k])).rjust(W)
            for k, _, _ in cols
        ))
    print()

    ref = next(r for r in rows if r["backend"] == "list")
    print(f"  dense equivalent: {ref['dense_mb']:.1f} MB "
          f"({ref['dense_mb'] / ref['payload_mb']:.2f}x the ragged payload)")
    for r in rows:
        if r["backend"] != "list":
            print(f"  {r['backend']:<8} vs list: "
                  f"memory {r['payload_mb'] / ref['payload_mb']:.2f}x, "
                  f"random access {r['get_500_ms'] / ref['get_500_ms']:.2f}x, "
                  f"to_dense {r['to_dense_ms'] / ref['to_dense_ms']:.2f}x, "
                  f"reduce {r['reduce_ms'] / ref['reduce_ms']:.2f}x")
        if "typestr" in r:
            print(f"  {r['backend']} layout: {r['typestr']}")
    return rows


if __name__ == "__main__":
    main()
