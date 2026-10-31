"""Date protocols.  A split is a list of dates, and the list is in the manifest.

Two family shapes exist:
  * ``dates``  -- the freeze file stores `train/val/test` date lists
  * ``folds``  -- the freeze file stores k blocks; VAL is the date range
                  `[val_first, val_last]`, TRAIN is everything else eligible

LaDe also has a `refrozen` arm: 22 quality-eligible weekdays instead of the
20-date file on disk.  Its dates are in `registry.LADE_REFROZEN`, not
recomputed, so a change in the audit code cannot silently move a TEST date.
"""
from __future__ import annotations

import json

from . import paths
from .registry import CHENGDU, LADE_REFROZEN


def _read(spec):
    return json.loads(paths.resolve(spec["freeze"]).read_text(encoding="utf-8"))


def _from_file(spec, which):
    d = _read(spec)
    sp = d["split"]
    if "folds" in sp:
        return dict(folds=[dict(block=[int(x) for x in f["block"]],
                                val_first=int(f["val_first"]),
                                val_last=int(f["val_last"]))
                           for f in sp["folds"]],
                    eligible=[int(x) for x in d["eligible_dates"]])
    if which == "frozen":
        return dict(train=[int(x) for x in sp["train"]["dates"]],
                    val=[int(x) for x in sp["val"]["dates"]],
                    test=[int(x) for x in sp["test"]["dates"]],
                    eligible=[int(x) for x in d.get("eligible_dates", [])])
    raise KeyError(f"{spec['name']}: no split {which!r} in the freeze file")


def resolve(spec, which="frozen", fold=0):
    """`dict(train, val, test)` for one object / split / fold."""
    name = spec["name"]
    if which not in spec["splits"]:
        raise KeyError(f"{name}: split {which!r} not in {spec['splits']}")

    if which == "window":                       # Chengdu daytime window
        return dict(train=list(CHENGDU["train"]), val=list(CHENGDU["val"]),
                    test=list(CHENGDU["test"]),
                    eligible=list(CHENGDU["dates"]),
                    hours=list(CHENGDU["hours"]))

    if which == "refrozen":                     # LaDe, quality-eligible
        return dict(train=list(LADE_REFROZEN["train"]),
                    val=list(LADE_REFROZEN["val"]),
                    test=list(LADE_REFROZEN["test"]),
                    eligible=list(LADE_REFROZEN["eligible"]))

    fz = _from_file(spec, which)
    if "folds" in fz:
        if not (0 <= int(fold) < len(fz["folds"])):
            raise KeyError(f"{name}: fold {fold} out of range "
                           f"(0..{len(fz['folds']) - 1})")
        f = fz["folds"][int(fold)]
        te = [int(x) for x in f["block"]]
        va = [x for x in fz["eligible"]
              if f["val_first"] <= x <= f["val_last"]]
        tr = [x for x in fz["eligible"]
              if x not in set(te) | set(va)]
        return dict(train=tr, val=va, test=te, eligible=fz["eligible"])
    return dict(train=fz["train"], val=fz["val"], test=fz["test"],
                eligible=fz["eligible"])


def all_splits(spec):
    """Every `(which, fold, split)` v1 can produce for this object."""
    out = []
    for which in spec["splits"]:
        if which == "frozen":
            n = spec["folds"]
            for f in range(n):
                out.append((which, f, resolve(spec, which, f)))
        else:
            out.append((which, 0, resolve(spec, which, 0)))
    return out
