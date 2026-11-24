# Plan clima v3 (modelo id=4 Climático+ONI) — 2026-11-24

Objetivo (autorizado por el usuario): que id 4 sea claramente mejor que v2 y que el naive+drift, sin perder su razón de ser (escenarios ENSO/clima).
Diagnóstico (ver `docs/work/04_clima/REPORTE.md`): v2 = deriva anual + β·ONI + reparto mensual promedio 2 años. No tiene memoria del nivel reciente
(ene-ago 2026: 5.07 vs 3.96 naive+drift; sesgo +ene-abr / −may-ago) y su sensibilidad ENSO bajó (Niño−Niña 9.7 % → 3.1 %).

## Ideas candidatas (el agente decide con evidencia, no por gusto)
1. Anclaje de nivel con info reciente al origen: usar los meses ya observados del año en curso / últimos 12 m (tasa de crecimiento reciente) en vez de solo totales de años cerrados.
2. Reparto mensual con tendencia intra-año (los shares promedio ignoran el crecimiento dentro del año) y/o modelo mensual log-yoy: log D_m = log D_{m-12} + g_reciente + β·ΔONI + γ·anomalía clima (temp, lluvia) con Ridge, ventana expansiva.
3. Efecto climático/ONI estimado sobre RESIDUOS de una base honesta (seasonal naive + drift) con regularización y signo físico validado; escenarios (Niño/Normal/Niña) deben seguir ordenados y con efecto no trivial (>= ~3 % anual Niño−Niña).
4. Combinación/shrinkage de drift reciente vs largo plazo (ponderación por recencia, hiperparámetro elegido solo en validación).

## Reglas (teoría ML, sin trampa)
- Walk-forward expansivo, cero fuga; features exactamente como en inferencia. Orígenes MENSUALES.
- Selección de hiperparámetros/variante SOLO con orígenes ≤ 2023-12 (objetivos ≤ 2025-12); 2024-2025 = validación; 2026 ene-ago = test final, se mira UNA vez al final. Nada ajustado mirando 2026.
- Comparar siempre contra: v2 actual, naive estacional+drift, y ids 1-3. Reportar MAPE h1-12/h13-24, sesgo, anual, y por periodo (≤2025 vs ≥2026). Sin reportar solo lo favorable.
- No hardcodear constantes por fecha/mes. Cap/override siguen siendo los de forecast_adjustments (no tocar).
- Contrato API intacto (campos y fechas start-of-month para id 4). Si se cambia `predict_future_climate`/artefactos, respaldar los actuales (`backup_modelos_originales`/v2) y mantener `ML_FORECAST_LEGACY`/compatibilidad.
- Tests nuevos en tests/forecast_climate; `python -m pytest tests` completo verde antes de reportar.
- Documentar: docs/work/09_clima_v3/REPORTE.md (decisiones, tablas, riesgos) + actualizar sessions/2026-11-23-mejora-modelos-forecast.md (sección clima v3).

## Estado
- [ ] implementación (agente-clima-v3)  - [ ] revisión multi-ángulo + regresión  - [ ] correcciones  - [ ] commit
