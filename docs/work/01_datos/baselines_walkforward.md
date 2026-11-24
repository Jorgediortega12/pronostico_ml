# Baselines walk-forward (piso a superar)

Generado por `docs/work/01_datos/run_baselines.py`. Orígenes 2017-12..2025-12, horizonte 24 meses,
expanding window. Train = serie limpia (dedupe 2020-Q4, 2021-09 imputado); real = solo meses observados
completos (2021-09 NO cuenta en métricas). Excluidos de métricas: COVID 2020-03..2021-02 y meses
≥ 2026-09 (quiebre de mercado, `demand_split_config` = 2026-09-01 / 100 %).

Errores en %: MAPE mensual, sesgo = media (pred−real)/real (negativo = subestima), `mape_anual` = error del
total de cada año calendario completo dentro del horizonte (`_h1_12` = año completo en h≤12).

## Resumen

| escenario          | modelo                 |   n_origenes |    n |   mape |   sesgo |   max_ape |   mape_h1_12 |   sesgo_h1_12 |   mape_h13_24 |   sesgo_h13_24 |   n_anual |   mape_anual |   sesgo_anual |   mape_anual_h1_12 |   mape_anual_h13_24 |
|:-------------------|:-----------------------|-------------:|-----:|-------:|--------:|----------:|-------------:|--------------:|--------------:|---------------:|----------:|-------------:|--------------:|-------------------:|--------------------:|
| origenes_mensuales | naive_estacional       |           97 | 1880 |   8.03 |   -7.21 |     23.98 |         6.27 |         -5.16 |         10.01 |          -9.53 |        66 |         6.92 |         -6.67 |               5.27 |                7.09 |
| origenes_mensuales | naive_estacional_drift |           97 | 1880 |   6.13 |    0.91 |     23.96 |         5.83 |          0.15 |          6.46 |           1.78 |        66 |         5.89 |          1.65 |               3.71 |                6.10 |
| origenes_diciembre | naive_estacional       |            9 |  170 |   8.14 |   -7.39 |     23.98 |         6.27 |         -5.25 |         10.31 |          -9.86 |        11 |         7.17 |         -6.79 |               5.27 |                9.45 |
| origenes_diciembre | naive_estacional_drift |            9 |  170 |   5.97 |    0.62 |     22.40 |         5.60 |          0.04 |          6.40 |           1.28 |        11 |         4.74 |          1.28 |               3.71 |                5.98 |

## Error del total anual por año objetivo (orígenes diciembre, h 1–12, %)

|   anio |   naive_estacional |   naive_estacional_drift |
|-------:|-------------------:|-------------------------:|
|   2018 |              -5.09 |                    -0.43 |
|   2019 |              -9.33 |                    -5.51 |
|   2022 |               2.10 |                    11.02 |
|   2023 |              -7.21 |                    -2.96 |
|   2024 |              -5.74 |                    -0.50 |
|   2025 |              -2.15 |                     1.86 |

## Diagnóstico del quiebre de mercado (informativo, no se usa para escalar)

| mes     |    GWh |   GWh_mismo_mes_anio_previo |   ratio_yoy |
|:--------|-------:|----------------------------:|------------:|
| 2026-09 | 587.18 |                      712.31 |        0.82 |
| 2026-10 | 576.89 |                      705.23 |        0.82 |

Dedupe: 92 fechas duplicadas resueltas (ver `dedup_log.csv`); razón max/min entre candidatos 8.2–11.8×.
