
## Sensibilidad a escenarios ONI (artefactos productivos v2, código real de inferencia)

| escenario                        |   total_2026_GWh |   total_2027_GWh |   crec_2026_% |   vs_NORMAL_2026_% |
|:---------------------------------|-----------------:|-----------------:|--------------:|-------------------:|
| Auto (ONI proyectado)            |          9094.23 |          9609.75 |          5.44 |              -0.01 |
| Todo NIÑA                        |          8981.7  |          9352.92 |          4.13 |              -1.25 |
| Todo NORMAL                      |          9095.52 |          9591.47 |          5.45 |               0    |
| Todo NIÑO                        |          9260.63 |          9942.85 |          7.37 |               1.82 |
| NIÑO desde may-2026 (resto auto) |          9209.7  |          9731.77 |          6.78 |               1.26 |


Un solo mes forzado (2026): demanda del mes por fase

| mes     |   NIÑA_GWh |   NORMAL_GWh |   NIÑO_GWh | orden_ok   |   spread_% |
|:--------|-----------:|-------------:|-----------:|:-----------|-----------:|
| 2026-01 |     735.87 |       736.64 |     742.75 | True       |       0.93 |
| 2026-02 |     676.33 |       677.04 |     691.13 | True       |       2.19 |
| 2026-03 |     761.4  |       762.2  |     763.34 | True       |       0.26 |
| 2026-04 |     747.72 |       748.5  |     757.85 | True       |       1.35 |
| 2026-05 |     775.45 |       776.26 |     777.43 | True       |       0.26 |
| 2026-06 |     732.05 |       750.47 |     751.6  | True       |       2.67 |
| 2026-07 |     795.95 |       798.72 |     799.94 | True       |       0.5  |
| 2026-08 |     786.02 |       789.25 |     790.44 | True       |       0.56 |
| 2026-09 |     750.83 |       761.87 |     769.49 | True       |       2.48 |
| 2026-10 |     764.93 |       766.6  |     770.92 | True       |       0.78 |
| 2026-11 |     730.21 |       730.98 |     741.73 | True       |       1.58 |
| 2026-12 |     785.07 |       785.9  |     796.76 | True       |       1.49 |


A posteriori (informativo, NO usado para elegir nada): error vs real ene–ago 2026 por escenario

|                                  |   MAPE_% |   sesgo_% |
|:---------------------------------|---------:|----------:|
| Auto (ONI proyectado)            |     5.07 |     -1.18 |
| Todo NIÑA                        |     4.66 |     -3.61 |
| Todo NORMAL                      |     3.68 |     -2.4  |
| Todo NIÑO                        |     3.41 |     -0.76 |
| NIÑO desde may-2026 (resto auto) |     4.63 |     -0.85 |
