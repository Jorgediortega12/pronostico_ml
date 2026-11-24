# Resultados nivel adaptativo (spec: {'k': 6, 'hl_w': 2.0, 'est': 'median', 'phi': 0.5, 'hl_h': 3.0, 'cap': 0.06, 'min_obs': 3, 'tau': 0.0, 'agree': 0.0})

## TUNE (todos los h 1-24)

| modelo          |    n |   MAPE sin |   MAPE con |   delta |   sesgo sin |   sesgo con |
|:----------------|-----:|-----------:|-----------:|--------:|------------:|------------:|
| RF (id 1)       | 1440 |       5.31 |       5.23 |   -0.08 |       -0.68 |       -0.61 |
| XGB (id 2)      | 1440 |       5.34 |       5.29 |   -0.05 |       -0.66 |       -0.59 |
| SARIMAX (id 3)  | 1440 |       4.04 |       4.04 |   -0    |        1.24 |        1.11 |
| Clima v3 (id 4) | 1440 |       3.91 |       3.88 |   -0.04 |        0.08 |        0.05 |
| naive+drift     | 1440 |       6.57 |     nan    |  nan    |        0.59 |      nan    |

## VAL (todos los h 1-24)

| modelo          |   n |   MAPE sin |   MAPE con |   delta |   sesgo sin |   sesgo con |
|:----------------|----:|-----------:|-----------:|--------:|------------:|------------:|
| RF (id 1)       | 276 |       6.67 |       6.23 |   -0.44 |        6.62 |        6.12 |
| XGB (id 2)      | 276 |       6.29 |       5.85 |   -0.44 |        6.22 |        5.75 |
| SARIMAX (id 3)  | 276 |       4.19 |       4.07 |   -0.13 |        4.04 |        3.77 |
| Clima v3 (id 4) | 276 |       3.32 |       3.19 |   -0.14 |        3.15 |        3    |
| naive+drift     | 276 |       4.24 |     nan    |  nan    |        3.2  |      nan    |

## VAL: MAPE por horizonte (sin -> con)

| modelo          |   h1-3 sin |   h1-3 con |   h4-6 sin |   h4-6 con |   h7-12 sin |   h7-12 con |   h13-24 sin |   h13-24 con |
|:----------------|-----------:|-----------:|-----------:|-----------:|------------:|------------:|-------------:|-------------:|
| RF (id 1)       |       4.93 |       3.78 |       6.25 |       5.67 |        7.24 |        7.08 |         8.01 |         8.03 |
| XGB (id 2)      |       5.47 |       4.3  |       6.15 |       5.57 |        6.76 |        6.62 |         6.62 |         6.64 |
| SARIMAX (id 3)  |       2.55 |       2.4  |       3.66 |       3.38 |        4.36 |        4.25 |         6.08 |         6.08 |
| Clima v3 (id 4) |       2.61 |       2.12 |       3.04 |       2.91 |        3.45 |        3.45 |         4.13 |         4.16 |

## VAL: por mes de inicio

| inicio                  | modelo          |   n |   MAPE sin |   MAPE con |   sesgo sin |   sesgo con |
|:------------------------|:----------------|----:|-----------:|-----------:|------------:|------------:|
| inicio ene (origen dic) | RF (id 1)       |  12 |       7.51 |       6.34 |        7.51 |        6.34 |
| inicio ene (origen dic) | XGB (id 2)      |  12 |       7.78 |       6.76 |        7.78 |        6.76 |
| inicio ene (origen dic) | SARIMAX (id 3)  |  12 |       2.88 |       2.55 |        2.86 |        1.86 |
| inicio ene (origen dic) | Clima v3 (id 4) |  12 |       2.8  |       2.41 |        2.77 |        2.07 |
| inicio ene (origen dic) | naive+drift     |  12 |       3.05 |     nan    |        1.89 |      nan    |
| inicio may (origen abr) | RF (id 1)       |  28 |       6.78 |       6.64 |        6.78 |        6.64 |
| inicio may (origen abr) | XGB (id 2)      |  28 |       5.96 |       5.79 |        5.96 |        5.74 |
| inicio may (origen abr) | SARIMAX (id 3)  |  28 |       4.77 |       4.81 |        4.77 |        4.81 |
| inicio may (origen abr) | Clima v3 (id 4) |  28 |       3.55 |       3.77 |        3.47 |        3.66 |
| inicio may (origen abr) | naive+drift     |  28 |       4.45 |     nan    |        3.84 |      nan    |
| inicio jul (origen jun) | RF (id 1)       |  24 |       6.28 |       5.85 |        6.28 |        5.64 |
| inicio jul (origen jun) | XGB (id 2)      |  24 |       5.62 |       5.1  |        5.62 |        4.89 |
| inicio jul (origen jun) | SARIMAX (id 3)  |  24 |       4    |       3.76 |        3.92 |        3.59 |
| inicio jul (origen jun) | Clima v3 (id 4) |  24 |       3.52 |       3.3  |        3.43 |        3.16 |
| inicio jul (origen jun) | naive+drift     |  24 |       4.33 |     nan    |        3.84 |      nan    |

## Por anio del origen (objetivos <= 2025-12)

|   origen anio |   n |   rf sin |   rf con |   xgb sin |   xgb con |   sarimax sin |   sarimax con |   clima sin |   clima con |
|--------------:|----:|---------:|---------:|----------:|----------:|--------------:|--------------:|------------:|------------:|
|          2017 |  24 |     6.7  |     6.59 |      6.58 |      6.46 |          2.35 |          2.24 |        2.31 |        2.31 |
|          2018 | 233 |     7.06 |     6.98 |      6.99 |      6.9  |          2.8  |          2.83 |        2.94 |        2.97 |
|          2019 | 141 |     4.3  |     3.83 |      3.72 |      3.35 |          3.65 |          3.63 |        4.18 |        4.08 |
|          2020 | 187 |     5.73 |     5.71 |      5.49 |      5.48 |          4.28 |          4.26 |        4.59 |        4.56 |
|          2021 | 279 |     6.18 |     6.36 |      6.21 |      6.38 |          5.6  |          5.69 |        5.27 |        5.25 |
|          2022 | 288 |     4.28 |     3.97 |      4.79 |      4.5  |          4.52 |          4.25 |        4.1  |        3.93 |
|          2023 | 288 |     4.18 |     4.25 |      4.31 |      4.46 |          3.23 |          3.42 |        2.77 |        2.8  |
|          2024 | 210 |     7.17 |     7.05 |      6.62 |      6.55 |          4.55 |          4.54 |        3.58 |        3.59 |
|          2025 |  66 |     5.08 |     3.63 |      5.26 |      3.63 |          3.07 |          2.57 |        2.52 |        1.92 |

## Tasa de origenes (MAPE del origen, sin -> con)

| periodo   | modelo          |   origenes |   mejora % |   empeora > 0.5pp % |
|:----------|:----------------|-----------:|-----------:|--------------------:|
| TUNE      | RF (id 1)       |         73 |       65.8 |                 6.8 |
| TUNE      | XGB (id 2)      |         73 |       67.1 |                 6.8 |
| TUNE      | SARIMAX (id 3)  |         73 |       52.1 |                 4.1 |
| TUNE      | Clima v3 (id 4) |         73 |       58.9 |                 0   |
| VAL       | RF (id 1)       |         21 |       76.2 |                 9.5 |
| VAL       | XGB (id 2)      |         21 |       81   |                 4.8 |
| VAL       | SARIMAX (id 3)  |         21 |       76.2 |                 4.8 |
| VAL       | Clima v3 (id 4) |         21 |       76.2 |                 9.5 |
