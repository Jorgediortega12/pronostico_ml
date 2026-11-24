# Sanidad de forma y nivel (2026-11-24)

Script: `sanidad_forma_nivel.py` (API real, modo crudo, horizonte 2026-01..2029-12, BD local). Historico real: 2019-2025 completo, 2026 solo ene-ago.

Referencia historica: anual real 2023 7.955 / 2024 8.440 / 2025 8.625 TWh; crecimiento 7.8 % / 6.1 % / 2.2 %. Perfil estacional 2023-25: pico julio, valle febrero, amplitud max/min 1.169.

| Modelo | Crec. 2026 / 2027 / 2028 / 2029 (%) | Corr. forma vs hist (2027 / 2028) | Pico / valle | Amplitud |
|---|---|---|---|---|
| RF | 6.3 / 6.1 / 6.4 / 5.1 | 0.979 / 0.959 | jul / feb | 1.17 / 1.12 |
| XGB | 6.6 / 5.9 / 5.5 / 5.1 | 0.980 / 0.966 | jul / feb | 1.15 / 1.12 |
| SARIMAX | 7.1 / 6.5 / 6.1 / 5.6 | 0.995 / 0.987 | jul / feb | 1.18 / 1.14 |
| Clima+ONI | 5.4 / 8.3 / 5.7 / 5.7 | 0.893 / 0.893 | jul / feb | 1.14 / 1.14 |

Conclusiones
- Forma: los 4 reproducen pico (julio), valle (febrero) y amplitud historicos; saltos mensuales dentro del rango historico (max +14.6 % / min -9.6 % vs +19.4 / -13.1 historico).
- Nivel: 2026 completo 9.09-9.24 TWh (+5.4 % a +7.1 % vs 2025), coherente con la tendencia 2022-2025 (media ~5.7 %), aunque por encima del +2.2 % de 2025. Ene-ago 2026 los modelos quedan 0-6 % por debajo del real en may-ago (sesgo a la baja reciente).
- Puntos a vigilar: (1) id 4 salta a +8.3 % en 2027 y repite la misma forma anual en 2027 y 2028 (modelo anual, por diseno; correlacion 0.893, la menor); (2) RF/XGB/SARIMAX a 2028-2029 siguen tendencia (+5-6 %/anio) y superan el maximo mensual historico (≈950-970 vs 848 GWh): extrapolacion de tendencia, no validable.
- Fix menor: `baseline.py` modo crudo ya no rompe (`override_aplicado`).
