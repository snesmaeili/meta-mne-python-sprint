"""A3-1 — `epochs[...]` sub-selection: do the per-epoch bounds follow?"""

import numpy as np

from a3_lifecycle import audit, banner, lengths_of, make, tags

LENGTHS = (100, 250, 75, 180, 120)
TMINS = (0.0, -0.2, 0.1, -0.5, 0.3)

FAILS = []


def check(label, ep, expect_tags, base_tmins=TMINS):
    exp_tmins = [base_tmins[t] for t in expect_tags]
    msgs = audit(ep, label, expect_tags=expect_tags, expect_tmins=exp_tmins)
    if msgs:
        FAILS.extend(msgs)
        for m in msgs:
            print("  FAIL", m)
    else:
        print(
            f"  ok   {label:38s} tags={tags(ep)} lens={lengths_of(ep)} "
            f"sel={list(ep.selection)}"
        )


def main():
    banner("copy=True path (epochs[item])")
    cases = [
        ("int 0", 0, [0]),
        ("int 3", 3, [3]),
        ("int -1", -1, [4]),
        ("int -3", -3, [2]),
        ("slice 1:4", slice(1, 4), [1, 2, 3]),
        ("slice ::2", slice(None, None, 2), [0, 2, 4]),
        ("slice ::-1", slice(None, None, -1), [4, 3, 2, 1, 0]),
        ("slice 3:0:-1", slice(3, 0, -1), [3, 2, 1]),
        ("slice -2:", slice(-2, None), [3, 4]),
        ("list [3,1]", [3, 1], [3, 1]),
        ("list [4,4,0]", [4, 4, 0], [4, 4, 0]),
        ("list [-1,-2]", [-1, -2], [4, 3]),
        ("bool mask", np.array([True, False, True, False, True]), [0, 2, 4]),
        ("bool all False", np.array([False] * 5), []),
        ("empty list", [], []),
        ("ndarray int", np.array([2, 0]), [2, 0]),
    ]
    for label, item, want in cases:
        ep = make(LENGTHS, TMINS)
        try:
            sub = ep[item]
        except Exception as exc:
            print(f"  RAISE {label:38s} {type(exc).__name__}: {exc}")
            FAILS.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        check(label, sub, want)
        # the parent must be untouched
        pmsgs = audit(ep, f"{label} PARENT", expect_tags=list(range(5)))
        if pmsgs:
            FAILS.extend(pmsgs)
            for m in pmsgs:
                print("  FAIL(parent)", m)

    banner("copy=False path (_getitem(..., copy=False))")
    for label, item, want in cases:
        ep = make(LENGTHS, TMINS)
        try:
            out = ep._getitem(item, copy=False)
        except Exception as exc:
            print(f"  RAISE {label:38s} {type(exc).__name__}: {exc}")
            FAILS.append(f"copy=False {label} raised {type(exc).__name__}: {exc}")
            continue
        if out is not ep:
            FAILS.append(f"copy=False {label}: returned a different object")
        check(f"cf {label}", ep, want)

    banner("by event_id name")
    ep = make(LENGTHS, TMINS, event_ids=[1, 2, 1, 2, 1])
    for label, item, want in [
        ("['e1']", "e1", [0, 2, 4]),
        ("['e2']", "e2", [1, 3]),
        ("['e1','e2']", ["e1", "e2"], [0, 1, 2, 3, 4]),
    ]:
        sub = ep[item]
        check(label, sub, want)

    banner("metadata query")
    pd = None
    try:
        import pandas as pd
    except ImportError:
        print("  pandas missing, skipped")
    if pd is not None:
        md = pd.DataFrame(dict(col=[0, 2, 4, 6, 8], name=list("abcde")))
        ep = make(LENGTHS, TMINS, metadata=md)
        for label, q, want in [
            ("col > 3", "col > 3", [2, 3, 4]),
            ("col > 100 (empty)", "col > 100", []),
            ("name == 'b'", "name == 'b'", [1]),
        ]:
            try:
                sub = ep[q]
            except Exception as exc:
                print(f"  RAISE {label:38s} {type(exc).__name__}: {exc}")
                FAILS.append(f"metadata {label}: {type(exc).__name__}: {exc}")
                continue
            check(label, sub, want)
            if sub.metadata is not None and len(sub.metadata):
                got = list(sub.metadata["col"])
                exp = [md["col"][t] for t in want]
                if got != exp:
                    FAILS.append(f"metadata {label}: col {got} != {exp}")

    banner("chained subselection")
    ep = make(LENGTHS, TMINS)
    a = ep[::2]  # 0,2,4
    b = a[::-1]  # 4,2,0
    c = b[1:]  # 2,0
    check("[::2][::-1][1:]", c, [2, 0])

    banner("SUMMARY")
    print(f"{len(FAILS)} violations")
    for f in FAILS:
        print("  -", f)


if __name__ == "__main__":
    main()
