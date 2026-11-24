# Resultados walk-forward modelo id=4 (generado por walkforward_clima.py --final)


## Orígenes dic — orígenes 2017-12..2025-12 (incluye 2026-01..08)

| modelo                                         |   n_origenes |   n |   mape |   sesgo |   max_ape |   mape_h1_12 |   mape_h13_24 |   mape_anual |   sesgo_anual |
|:-----------------------------------------------|-------------:|----:|-------:|--------:|----------:|-------------:|--------------:|-------------:|--------------:|
| antes_artefacto_prod (in-sample <=2025)        |            9 | 170 |   7.92 |    4.36 |     28.1  |         7.59 |          8.29 |         5.46 |          5.46 |
| antes_receta_original (reentrenada por origen) |            9 | 170 |   9.55 |   -7.54 |     36.58 |         8.57 |         10.67 |         8.23 |         -7.12 |
| receta_original + datos_limpios                |            9 | 170 |   8.67 |   -8.34 |     21.43 |         7.53 |          9.99 |         8.21 |         -8.21 |
| despues_v2 (nys=2, sin parche)                 |            9 | 170 |   4.55 |    0.38 |     16.44 |         4.59 |          4.51 |         3.39 |          0.99 |
| despues_v2 (nys=5, con parche)                 |            9 | 170 |   4.4  |    0.36 |     16.24 |         4.17 |          4.66 |         3.39 |          0.99 |
| solo_deriva (mismo spec sin ONI)               |            9 | 170 |   4.56 |    0.92 |     17.34 |         4.69 |          4.41 |         3.46 |          1.5  |
| naive_estacional                               |            9 | 170 |   8.14 |   -7.39 |     23.98 |         6.27 |         10.31 |         7.17 |         -6.79 |
| naive_estacional_drift                         |            9 | 170 |   5.97 |    0.62 |     22.4  |         5.6  |          6.4  |         4.74 |          1.28 |


## Orígenes dic — solo objetivos <= 2025-12 (sin test)

| modelo                                         |   n_origenes |   n |   mape |   sesgo |   max_ape |   mape_h1_12 |   mape_h13_24 |   mape_anual |   sesgo_anual |
|:-----------------------------------------------|-------------:|----:|-------:|--------:|----------:|-------------:|--------------:|-------------:|--------------:|
| antes_artefacto_prod (in-sample <=2025)        |            8 | 154 |   8.28 |    5.01 |     28.1  |         7.81 |          8.83 |         5.46 |          5.46 |
| antes_receta_original (reentrenada por origen) |            8 | 154 |  10.07 |   -8.08 |     36.58 |         8.89 |         11.45 |         8.23 |         -7.12 |
| receta_original + datos_limpios                |            8 | 154 |   9.03 |   -8.83 |     21.43 |         7.74 |         10.54 |         8.21 |         -8.21 |
| despues_v2 (nys=2, sin parche)                 |            8 | 154 |   4.57 |    0.35 |     16.44 |         4.54 |          4.6  |         3.39 |          0.99 |
| despues_v2 (nys=5, con parche)                 |            8 | 154 |   4.39 |    0.34 |     16.24 |         4.15 |          4.67 |         3.39 |          0.99 |
| solo_deriva (mismo spec sin ONI)               |            8 | 154 |   4.57 |    0.87 |     17.34 |         4.65 |          4.48 |         3.46 |          1.5  |
| naive_estacional                               |            8 | 154 |   8.19 |   -7.4  |     23.98 |         6.2  |         10.51 |         7.17 |         -6.79 |
| naive_estacional_drift                         |            8 | 154 |   6.06 |    0.77 |     22.4  |         5.75 |          6.42 |         4.74 |          1.28 |


## Orígenes mens — orígenes 2017-12..2025-12 (incluye 2026-01..08)

| modelo                                         |   n_origenes |    n |   mape |   sesgo |   max_ape |   mape_h1_12 |   mape_h13_24 |   mape_anual |   sesgo_anual |
|:-----------------------------------------------|-------------:|-----:|-------:|--------:|----------:|-------------:|--------------:|-------------:|--------------:|
| antes_artefacto_prod (in-sample <=2025)        |           97 | 1880 |   9.26 |    6.06 |     31.62 |         8.1  |         10.57 |         5.82 |          5.82 |
| antes_receta_original (reentrenada por origen) |           97 | 1880 |  11.1  |   -8.62 |     47.56 |        10.15 |         12.18 |         9.12 |         -6.89 |
| receta_original + datos_limpios                |           97 | 1880 |  10.22 |  -10.01 |     25.88 |         8.81 |         11.83 |         9.88 |         -9.88 |
| despues_v2 (nys=2, sin parche)                 |           97 | 1880 |   4.52 |    0.51 |     16.75 |         4.78 |          4.24 |         3.64 |          1.02 |
| despues_v2 (nys=5, con parche)                 |           97 | 1880 |   4.52 |    0.48 |     16.56 |         4.76 |          4.25 |         3.64 |          1.02 |
| solo_deriva (mismo spec sin ONI)               |           97 | 1880 |   4.57 |    1.19 |     18.58 |         4.72 |          4.39 |         3.74 |          1.74 |
| naive_estacional                               |           97 | 1880 |   8.03 |   -7.21 |     23.98 |         6.27 |         10.01 |         6.92 |         -6.67 |
| naive_estacional_drift                         |           97 | 1880 |   6.13 |    0.91 |     23.96 |         5.83 |          6.46 |         5.89 |          1.65 |


## Orígenes mens — solo objetivos <= 2025-12 (sin test)

| modelo                                         |   n_origenes |    n |   mape |   sesgo |   max_ape |   mape_h1_12 |   mape_h13_24 |   mape_anual |   sesgo_anual |
|:-----------------------------------------------|-------------:|-----:|-------:|--------:|----------:|-------------:|--------------:|-------------:|--------------:|
| antes_artefacto_prod (in-sample <=2025)        |           96 | 1716 |   9.77 |    6.57 |     31.62 |         8.42 |         11.37 |         5.82 |          5.82 |
| antes_receta_original (reentrenada por origen) |           96 | 1716 |  11.7  |   -9.11 |     47.56 |        10.63 |         12.96 |         9.12 |         -6.89 |
| receta_original + datos_limpios                |           96 | 1716 |  10.54 |  -10.36 |     25.88 |         9.12 |         12.22 |         9.88 |         -9.88 |
| despues_v2 (nys=2, sin parche)                 |           96 | 1716 |   4.59 |    0.31 |     16.75 |         4.8  |          4.34 |         3.64 |          1.02 |
| despues_v2 (nys=5, con parche)                 |           96 | 1716 |   4.54 |    0.28 |     16.56 |         4.72 |          4.32 |         3.64 |          1.02 |
| solo_deriva (mismo spec sin ONI)               |           96 | 1716 |   4.63 |    1.01 |     18.58 |         4.71 |          4.52 |         3.74 |          1.74 |
| naive_estacional                               |           96 | 1716 |   8.08 |   -7.23 |     23.98 |         6.28 |         10.2  |         6.92 |         -6.67 |
| naive_estacional_drift                         |           96 | 1716 |   6.2  |    1.01 |     23.96 |         6    |          6.42 |         5.89 |          1.65 |


## Test final 2026 (origen 2025-12, ene–ago 2026, crudo, una sola vez)

| modelo                                         |   mape |   sesgo |   max_ape |   n |   err_total_ene_ago |
|:-----------------------------------------------|-------:|--------:|----------:|----:|--------------------:|
| antes_artefacto_prod (in-sample <=2025)        |   5.28 |   -3.08 |      9.83 |   8 |               -3.48 |
| antes_receta_original (reentrenada por origen) |   5.28 |   -3.08 |      9.83 |   8 |               -3.48 |
| despues_v2 (nys=2, sin parche)                 |   5.07 |   -1.18 |      8.45 |   8 |               -1.58 |
| despues_v2 (nys=5, con parche)                 |   4.45 |   -1.7  |      8.03 |   8 |               -2.07 |
| naive_estacional                               |   6.97 |   -6.8  |     11.97 |   8 |               -7.11 |
| naive_estacional_drift                         |   3.96 |   -1.63 |      7.08 |   8 |               -1.96 |
| receta_original + datos_limpios                |   5.31 |   -3.26 |      9.99 |   8 |               -3.65 |
| solo_deriva (mismo spec sin ONI)               |   5.11 |   -0.36 |      9.35 |   8 |               -0.77 |


Error % por mes 2026:

| objetivo            |   antes_artefacto_prod (in-sample <=2025) |   antes_receta_original (reentrenada por origen) |   despues_v2 (nys=2, sin parche) |   despues_v2 (nys=5, con parche) |   naive_estacional |   naive_estacional_drift |   receta_original + datos_limpios |   solo_deriva (mismo spec sin ONI) |
|:--------------------|------------------------------------------:|-------------------------------------------------:|---------------------------------:|---------------------------------:|-------------------:|-------------------------:|----------------------------------:|-----------------------------------:|
| 2026-01-01 00:00:00 |                                      1.24 |                                             1.24 |                             3.22 |                             1.46 |              -4.08 |                     1.24 |                              1.05 |                               4.08 |
| 2026-02-01 00:00:00 |                                      6.37 |                                             6.37 |                             8.45 |                             7.85 |               0.66 |                     6.25 |                              6.17 |                               9.35 |
| 2026-03-01 00:00:00 |                                      1.18 |                                             1.18 |                             3.17 |                             1.67 |              -3.64 |                     1.71 |                              1    |                               4.02 |
| 2026-04-01 00:00:00 |                                     -1.21 |                                            -1.21 |                             0.73 |                            -0.69 |              -5.14 |                     0.12 |                             -1.39 |                               1.56 |
| 2026-05-01 00:00:00 |                                     -7.99 |                                            -7.99 |                            -6.18 |                            -5.58 |             -11.35 |                    -6.44 |                             -8.15 |                              -5.4  |
| 2026-06-01 00:00:00 |                                     -7.59 |                                            -7.59 |                            -5.78 |                            -5.38 |              -9.5  |                    -4.48 |                             -7.76 |                              -5    |
| 2026-07-01 00:00:00 |                                     -6.81 |                                            -6.81 |                            -4.98 |                            -4.93 |              -9.4  |                    -4.38 |                             -6.98 |                              -4.19 |
| 2026-08-01 00:00:00 |                                     -9.83 |                                            -9.83 |                            -8.06 |                            -8.03 |             -11.97 |                    -7.08 |                             -9.99 |                              -7.3  |


## Corrector post-proceso calibrado sin fuga (walk-forward)

| alcance              |   mape_crudo |   mape_corregido |   sesgo_crudo |   sesgo_corregido |   n |
|:---------------------|-------------:|-----------------:|--------------:|------------------:|----:|
| objetivos <= 2025-12 |         4.57 |             5.6  |          0.35 |              0.68 | 154 |
| test 2026            |         5.07 |             5.86 |         -1.18 |             -2.47 |   8 |
