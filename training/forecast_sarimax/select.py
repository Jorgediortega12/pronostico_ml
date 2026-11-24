"""Selección walk-forward (orígenes 2017-12..2023-12, h=24, reales <= 2025-12; 2026 NO se usa).

Uso:
    python -m training.forecast_sarimax.select --stage 2
    python -m training.forecast_sarimax.select --stage 3 --base <spec_name> [<spec_name> ...]
    python -m training.forecast_sarimax.select --stage ens --members <spec_name> <spec_name> ...
Salida: docs/work/03_sarimax/select_<stage>_resumen.csv y select_<stage>_detalle.csv.gz
"""
from __future__ import annotations

import argparse

import pandas as pd

from training.forecast_sarimax import candidates
from training.forecast_sarimax.wf import (
    HORIZON,
    OUT,
    SELECTION_ORIGINS,
    EnsembleFn,
    OriginalFn,
    SpecFn,
    baseline_runs,
    default_exclusions,
    load_data,
    make_origins,
    parallel_walk_forward,
    summarize,
)

COLS = ["modelo", "n", "mape", "sesgo", "max_ape", "mape_h1_12", "mape_h13_24", "mape_anual", "sesgo_anual"]


def run(cands: dict, tag: str, include_original: bool = False) -> pd.DataFrame:
    train, act_dev, _, mb = load_data()
    origins = make_origins(*SELECTION_ORIGINS)
    runs = baseline_runs(train, origins, HORIZON, act_dev)
    if include_original:
        runs.append(parallel_walk_forward(OriginalFn(True), train, origins, HORIZON, act_dev, "original_516_macro"))
        runs.append(parallel_walk_forward(OriginalFn(False), train, origins, HORIZON, act_dev, "original_516_sinmacro"))
    for name, fn in cands.items():
        print("->", name, flush=True)
        runs.append(parallel_walk_forward(fn, train, origins, HORIZON, act_dev, name))
    res = pd.concat(runs, ignore_index=True)
    summ = summarize(res, default_exclusions(mb)).sort_values("mape")
    res.to_csv(OUT / f"select_{tag}_detalle.csv.gz", index=False)
    summ.to_csv(OUT / f"select_{tag}_resumen.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(summ[COLS].round(3).to_string(index=False))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--base", nargs="*", default=[])
    ap.add_argument("--members", nargs="*", default=[])
    args = ap.parse_args()
    all_specs = {**candidates.stage2(), **candidates.stage2b()}
    if args.stage == "2":
        run({k: SpecFn(v) for k, v in candidates.stage2().items()}, "s2", include_original=True)
    elif args.stage == "2b":
        run({k: SpecFn(v) for k, v in candidates.stage2b().items()}, "s2b")
    elif args.stage == "3":
        base = [all_specs[b] for b in args.base]
        run({k: SpecFn(v) for k, v in candidates.stage3(base).items()}, "s3")
    elif args.stage == "ens":
        pool = {**all_specs}
        for b in list(all_specs.values()):
            pool.update(candidates.stage3([b]))
        members = [pool[m] for m in args.members]
        name = "ens[" + " | ".join(args.members) + "]"
        import hashlib

        run({name: EnsembleFn(members)}, "ens_" + hashlib.md5(name.encode()).hexdigest()[:8])
    else:
        raise SystemExit("stage desconocida")


if __name__ == "__main__":
    main()
