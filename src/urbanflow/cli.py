"""Command line: `python -m urbanflow_ref <command>`.

    manifest          print the pinned payload (or just its digest)
    describe NAME     what a citation to this object resolves to
    split NAME        the TRAIN / VAL / TEST dates actually used
    evaluate NAME     run the chain; `--out` writes the full record
    build-manifest    hash the inputs and freeze a new manifest
    verify            the regression suite

Every command that computes prints the reference id and digest first, so a
number pasted out of the terminal carries its provenance with it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from . import manifest as MF
from . import paths, registry, selfcheck
from ._version import MANIFEST_DIGEST, REFERENCE_ID
from .reference import Reference


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _load(args, verify=True):
    p = getattr(args, "manifest", None)
    return Reference.load(p, verify=verify)


def _goldens_path(args):
    p = getattr(args, "goldens", None)
    if p:
        return p
    d = paths.package_dir() / "goldens" / "urbanflow_reference_v1.json"
    return d if d.exists() else None


# --------------------------------------------------------------------------
def cmd_manifest(args):
    ref = _load(args, verify=False)
    if args.digest_only:
        print(ref.digest)
        return 0
    body = ref.manifest if not args.section else ref.manifest[args.section]
    print(json.dumps(jsonable(body), indent=1, sort_keys=True,
                     ensure_ascii=False))
    return 0


def cmd_describe(args):
    ref = _load(args)
    d = ref.describe(args.name)
    if args.json:
        print(json.dumps(jsonable(d), indent=1, sort_keys=True,
                         ensure_ascii=False))
        return 0
    sp = d["first_split"]
    print(f"{d['reference']['id']}  {d['reference']['digest'][:16]}")
    print(f"object      {args.name}  ({d['object']['family']}, "
          f"{d['object']['phases']} phases/day)")
    c = d["field"]["cache"]
    print(f"field       {c['path']}")
    print(f"            sha256 {c['sha256'][:16]}  {c['bytes']} B  "
          f"{c['n_windows']} windows x {c['grid'][0]}x{c['grid'][1]} grid  "
          f"dt={c['dt_min']} min")
    print(f"freeze      {d['data']['freeze']['path']}")
    b = d["road_basis"]
    print(f"basis       {b['path']}")
    print(f"            {b['n_int']} faces x r={b['r']}  "
          f"sha256 {b['sha256'][:16]}")
    print(f"operator    x = D F ; F0 = P0 F ; y = U^T F0 ; ridge, "
          f"lam = kappa*s_max, kappa on VAL")
    print(f"metric      E = sum||v-v^||^2 / sum||v||^2 ; score = 1-E ; "
          f"frames P/D/I/CLIM")
    if sp:
        print(f"split       {len(sp['train'])}/{len(sp['val'])}/"
              f"{len(sp['test'])} dates  "
              f"TRAIN {sp['train'][0]}..{sp['train'][-1]}  "
              f"TEST {sp['test']}")
    return 0


def cmd_split(args):
    ref = _load(args)
    sp = ref.split(args.name, args.split, args.fold)
    print(json.dumps(jsonable(sp), indent=1))
    return 0


def cmd_evaluate(args):
    ref = _load(args)
    r = ref.evaluate(args.name, args.split, args.fold)
    f, e, c = r["frames"], r["estimates"], r["counts"]
    print(f"reference   {r['reference']['id']}  "
          f"{r['reference']['digest'][:16]}")
    print(f"object      {args.name}  split={args.split} fold={args.fold}")
    print(f"rows        train {c['n_train']}  val {c['n_val']}  "
          f"test {c['n_test']}  ({c['n_dates']} TEST dates)")
    print(f"kappa       D/I/CLIM {e['kappa']:g}   P {e['kappa_P']:g}   "
          f"df_eff {e['df_eff']:.2f}")
    print(f"R_P         {f['P']:.6f}")
    print(f"R_D         {f['D']:.6f}")
    print(f"R_CLIM      {f['CLIM']:.6f}")
    print(f"R_I         {f['I']:.6f}")
    print(f"dR_I        {f['dR_I']:+.6f}")
    print(f"identity    |lhs-rhs| = {r['identity']['abs_err']:.2e}")
    if r["per_date"]:
        vals = [v["dR_I"] for v in r["per_date"].values()]
        print(f"per-date    dR_I min {min(vals):+.4f}  max {max(vals):+.4f}  "
              f"n>0 {sum(v > 0 for v in vals)}/{len(vals)}")
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(jsonable(r), indent=1, sort_keys=True),
                     encoding="utf-8")
        print(f"wrote       {p}")
    return 0


def cmd_build(args):
    objs = (tuple(args.objects.split(",")) if args.objects
            else registry.ALL_OBJECTS)
    print(f"building {REFERENCE_ID} over {len(objs)} objects ...", flush=True)
    payload = MF.build(objs, with_splits=not args.no_splits)
    out = pathlib.Path(args.out or (paths.package_dir() / "manifest.json"))
    digest = MF.save(payload, out)
    print(f"manifest    {out}")
    print(f"digest      {digest}")
    if digest != MANIFEST_DIGEST:
        print(f"note        _version.MANIFEST_DIGEST is {MANIFEST_DIGEST}; "
              f"update it to freeze this build", file=sys.stderr)
    if args.goldens:
        ref = Reference.load(out, verify=True, allow_unpinned=True)
        g = selfcheck.make_goldens(ref, progress=True)
        gp = pathlib.Path(args.goldens)
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(json.dumps(g, indent=1, sort_keys=True),
                      encoding="utf-8")
        print(f"goldens     {gp}  ({len(g['objects'])} splits)")
    return 0


def cmd_verify(args):
    ref = _load(args)
    rep = selfcheck.run(ref, golden_path=_goldens_path(args),
                        only=args.only, tol=args.tol,
                        legacy=not args.no_legacy)
    for c in rep["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"[{mark}] {c['name']:<18} {c['detail']}")
    print(f"\n{rep['passed']} passed, {rep['failed']} failed  "
          f"({ref.id} {ref.digest[:16]})")
    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(jsonable(rep), indent=1), encoding="utf-8")
    return 0 if rep["ok"] else 1


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m urbanflow_ref",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=None,
                    help="path to manifest.json (default: packaged)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest")
    p.add_argument("--digest-only", action="store_true")
    p.add_argument("--section", default=None)
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("describe")
    p.add_argument("name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_describe)

    p = sub.add_parser("split")
    p.add_argument("name")
    p.add_argument("--split", default="frozen")
    p.add_argument("--fold", type=int, default=0)
    p.set_defaults(fn=cmd_split)

    p = sub.add_parser("evaluate")
    p.add_argument("name")
    p.add_argument("--split", default="frozen")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("build-manifest")
    p.add_argument("--out", default=None)
    p.add_argument("--objects", default=None)
    p.add_argument("--goldens", default=None)
    p.add_argument("--no-splits", action="store_true")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("verify")
    p.add_argument("--goldens", default=None)
    p.add_argument("--only", default=None)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--no-legacy", action="store_true")
    p.add_argument("--json-out", default=None)
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
