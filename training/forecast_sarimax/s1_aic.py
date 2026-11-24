"""Etapa 1: rejilla de órdenes SARIMA por AIC/BIC con train <= 2023-12 (sin mirar validación ni test).

Uso: ``python training/forecast_sarimax/s1_aic.py`` → docs/work/03_sarimax/s1_aic.csv
"""
from __future__ import annotations

import itertools
import warnings

import pandas as pd
from joblib import Parallel, delayed

from training.forecast_sarimax.wf import AIC_TRAIN_END, N_JOBS, OUT, ModelSpec, load_data
from api.services.forecast_sarimax import fit_spec


def _fit(train, spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = fit_spec(train, spec)
            return {"spec": spec.name(), "order": spec.order, "sorder": spec.seasonal_order[:3], "trend": spec.trend,
                    "transform": spec.transform, "covid": spec.covid, "aic": r.aic, "bic": r.bic,
                    "k": len(r.params), "converged": bool(r.mle_retvals.get("converged", True))}
        except Exception as exc:  # noqa: BLE001
            return {"spec": spec.name(), "error": repr(exc)}


def main():
    train, *_ = load_data()
    train = train.loc[:AIC_TRAIN_END]
    specs = []
    for transform, covid in [("log", "nan"), ("log_daily", "nan")]:
        for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2)):
            specs.append(ModelSpec(order=(p, 1, q), seasonal_order=(P, 1, Q, 12), transform=transform, covid=covid))
        # d=0 con constante (tendencia determinista en la diferencia estacional)
        for p, q, Q in itertools.product((1, 2), (0, 1), (0, 1)):
            specs.append(ModelSpec(order=(p, 0, q), seasonal_order=(0, 1, Q, 12), trend="c", transform=transform, covid=covid))
    rows = Parallel(n_jobs=N_JOBS)(delayed(_fit)(train, s) for s in specs)
    df = pd.DataFrame(rows).sort_values(["transform", "bic"])
    df.to_csv(OUT / "s1_aic.csv", index=False)
    for tr, g in df.groupby("transform"):
        print(tr)
        print(g.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
