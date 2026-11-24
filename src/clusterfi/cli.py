"""clusterfi command line: bench | scale | report | demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _demo(args) -> None:
    from .clustering import ClusterConfig
    from .data import GenConfig, make_splits
    from .model import ModelConfig
    from .pipeline import run_methods, score_methods

    cfg = GenConfig(rho=args.rho, noise_frac=args.noise_frac, r2_latent=0.25)
    s = make_splits(cfg, args.seed)
    out = run_methods(s.train, s.valid, ModelConfig(), ClusterConfig(), args.seed, 3)
    K = s.truth.n_informative_groups
    rows = score_methods(out, s.truth, s.train.X, K)
    print(f"p={s.train.X.shape[1]} features, {len(s.truth.group_kind)} true groups, "
          f"{out.clustering.k} clusters found, validation R^2 {out.valid_r2:.3f}, k={K}")
    print(f"{'method':<8} {'group rc':>9} {'feat rc':>8} {'recall@k':>9} {'noise@k':>8}")
    for m, r in rows.items():
        print(f"{m:<8} {r['rank_corr']:>9.2f} {r['feature_rank_corr']:>8.2f} "
              f"{r['recall_at_k']:>9.2f} {r['noise_fdr_at_k']:>8.2f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="clusterfi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="full deterministic benchmark -> results/*.json, RESULTS.md")
    b.add_argument("--out", default="results")
    b.add_argument("--n-jobs", type=int, default=-1)
    s = sub.add_parser("scale", help="parallel permutation importance timing (machine-specific)")
    s.add_argument("--out", default="results")
    s.add_argument("--workers", default="1,2,4,8")
    s.add_argument("--n-valid", type=int, default=4000)
    s.add_argument("--passes", type=int, default=2)
    s.add_argument("--no-write", action="store_true", help="print only, do not write scale.json")
    r = sub.add_parser("report", help="re-render RESULTS.md from the JSON files")
    r.add_argument("--out", default="results")
    d = sub.add_parser("demo", help="one dataset, every method, printed")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--rho", type=float, default=0.7)
    d.add_argument("--noise-frac", type=float, default=0.5)
    args = ap.parse_args(argv)

    if args.cmd == "bench":
        from .bench import run
        run(Path(args.out), n_jobs=args.n_jobs)
    elif args.cmd == "scale":
        from .bench import render
        from .scale import run_scale, write_scale
        workers = tuple(int(w) for w in args.workers.split(","))
        kw = dict(workers=workers, n_valid=args.n_valid, passes=args.passes)
        res = run_scale(**kw) if args.no_write else write_scale(Path(args.out), **kw)
        print(json.dumps(res["rows"], indent=1))
        if not args.no_write and (Path(args.out) / "base.json").exists():
            (Path(args.out) / "RESULTS.md").write_text(render(Path(args.out)), encoding="utf-8",
                                                       newline="\n")
    elif args.cmd == "report":
        from .bench import render
        (Path(args.out) / "RESULTS.md").write_text(render(Path(args.out)), encoding="utf-8",
                                                   newline="\n")
    elif args.cmd == "demo":
        np.set_printoptions(precision=3)
        _demo(args)


if __name__ == "__main__":
    main()
