# REPORTE agente-datos (capa común + walk-forward)
Código: `training/common/{config,data,walkforward}.py`; tests: `tests/common/` (132 passed en suite completa).

## Uso
```python
from training.common.data import load_monthly_demand, target_series
from training.common.walkforward import make_origins, evaluate_with_baselines, default_exclusions
monthly = load_monthly_demand(impute_incomplete=True); mb = monthly.attrs["market_break"]
train = target_series(monthly); actuals = target_series(monthly, include_imputed=False)
res, summ = evaluate_with_baselines(fit_predict, train, make_origins("2017-12","2025-12"), 24,
            actuals=actuals, exclusions=default_exclusions(mb), model_name="x")
```
`fit_predict(train, h)` recibe solo la serie hasta el origen (`train.attrs["origen"]`); exógenas las filtra el modelo.
Modelo supera el piso si baja de naive+drift en MAPE mensual y en h1-12 sobre los mismos orígenes.

## Piso naive (orígenes 2017-12..2025-12, h=24, sin COVID ni >=2026-09)
| Orígenes | Baseline | MAPE | Sesgo | h1-12 | h13-24 | Anual h1-12 |
|---|---|---|---|---|---|---|
| mensuales (97) | naive estacional | 8.03 | -7.21 | 6.27 | 10.01 | 5.27 |
| mensuales (97) | naive+drift | 6.13 | +0.91 | 5.83 | 6.46 | 3.71 |
| diciembre (9) | naive+drift | 5.97 | +0.62 | 5.60 | 6.40 | 3.71 |

## Decisiones/hallazgos
- Fuente `total` (== suma 24 horas). Dedupe 2020-Q4: gana la fila con menor |log(total/ref)| (92 fechas; escalas 8-12x).
- Imputación solo 2021-09 (575.5 -> 616.6 GWh), solo para train. 2026-11 preliminar, nunca imputado.
- BD local: split 2026-09-01/100%; datos sep-oct 2026 = 0.82x del año previo (factor implícito ~0.74) — solo informativo.
- Constantes de dominio en config.py.
