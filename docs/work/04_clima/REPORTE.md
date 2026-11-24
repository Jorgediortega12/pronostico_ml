# REPORTE agente-clima — id=4 Climático+ONI v2

> **Nota de honestidad (ronda 2, docs/work/07_correcciones_r2).** Solo los objetivos >= 2026 (n=164) son fuera de
> muestra pura: v2 **3.83**, sin ONI 3.93, receta original reentrenada 4.88, naive+drift 5.40. En objetivos
> <= 2025-12 (n=276, ya vistos al seleccionar en la grilla de 288 specs) la receta original da 2.72 vs 4.06 de v2:
> **v2 no mejora a la original fuera de la selección salvo en los objetivos >= 2026** (riesgo de sobreajuste de la
> grilla). Además la API **no reentrena el id 4 en hindcast**: con start <= 2025 sus coeficientes vieron datos
> hasta 2025 (parcialmente in-sample); el error mostrado en hindcast solo es fuera de muestra para ids 1-3.
Datos limpios (2020-Q4 615/597/631 GWh; 2021-09 imputado; hasta 2026-08). Anual: `log1p(D_t)=log1p(D_{t-1})+mu+beta*oni_mean_t` (mu=5.49%/año de últimos 5 años utilizables sin 2020-22; beta=1.80% por unidad ONI, >=0). Reparto mensual: promedio 2 años (contrato intacto; es lo que usa la API tras 06_correcciones).
**Lectura honesta (06_correcciones):** es "deriva + sensibilidad ONI no validada": la variante sin ONI ("solo_deriva")
empata (WF total 97 orígenes 4.57 vs 4.52; post-selección 3.91 vs 3.97, mejor sin ONI). En los orígenes
posteriores a la selección (2024-01..2025-12, n=440) la receta original reentrenada por origen (3.52) le gana a v2
(3.97); naive+drift 4.67. La sensibilidad ENSO anual (Niño vs Niña) baja de ~9.7 % a ~3.1 % (aceptado). Corrector postproc recalibrado sin fuga: empeora (5.60 vs 4.57) -> no se aplica.
Selección: grid de 288 specs, orígenes <=2023-12, objetivos <=2025-12, regla fijada antes de ver 2026.

## Walk-forward (orígenes dic 2017-12..2025-12, crudo)
| Modelo | MAPE | Sesgo | h1-12 | h13-24 | Anual |
|---|---|---|---|---|---|
| Original reentrenado por origen | 9.55 | -7.54 | 8.57 | 10.67 | 8.23 |
| v2 | 4.55 | +0.38 | 4.59 | 4.51 | 3.39 |
| v2 + parche nys=5 (DESCARTADO) | 4.40 | +0.36 | 4.17 | 4.66 | 3.39 |
| naive+drift | 5.97 | +0.62 | 5.60 | 6.40 | 4.74 |
Orígenes mensuales (97): v2 4.52 = v2+nys5 4.52 vs sin ONI 4.57 vs naive+drift 6.13.
Orígenes posteriores a la selección 2024-01..2025-12 (n=440): v2 (nys=2) 3.97 | nys=5 4.22 | sin ONI 3.91 |
receta original reentrenada 3.52 | naive+drift 4.67. => nys=5 se descarta (empata en total, peor post-selección).

## Test final 2026 ene-ago (una vez)
Original 5.28 (sesgo -3.08) | v2 5.07 (-1.18) (lo que sirve la API) | v2+nys5 4.45 (descartado: un solo origen de 8 meses no justifica el cambio) | naive+drift 3.96. En 2026 naive+drift le gana a v2 (salto de demanda desde mayo no anticipable; ene-abr sobreestimado, may-ago subestimado -5 a -8%). Con escenario "Todo NIÑO" (a posteriori) MAPE 3.41.

## Escenarios ONI (2026): Niña 8982 (-1.25%) < Normal 9096 < Niño 9261 (+1.82%) GWh; orden correcto en los 12 meses.

## Decisiones / riesgos
- Test `test_predict_future_climate_single_month_sensitivity_spread_is_meaningful`: umbral 2.0% -> 0.5%. Justificación: el 2.9% previo venía del 2020-Q4 corrupto; con datos limpios el ajuste da 0.71% y el bug protegido 0.30%, 0.5% aún los separa. ACEPTADO por orquestador; margen estrecho (revisar en la ronda de revisión).
- oni_decay_v1.json calibrado con ONI 1950-2026 (fuga mínima de un parámetro físico, aceptada).
- Parche nys=5: se aplicó en la integración y se REVIRTIÓ en 06_correcciones (forecast.py usa n_years_shape=2;
  model_info.pkl `n_years_shape_recomendado` = 2). Registro: `docs/work/04_clima/forecast_clima_patch.md`.
- id 4 con año de inicio > último año del histórico + 1 (hoy 2028+): error explícito (antes etiquetaba mal).
Reproducir: `python -m training.forecast_climate.walkforward_clima --tune|--final|--escenarios` y `python -m training.forecast_climate.train_final_clima`.
