# Plan: corrección adaptativa de nivel (ids 1 RF, 2 XGB, 4 Clima v3; evaluar también id 3) — 2026-11-24

Problema (docs/work/11 / diagnóstico en sesión): desde mayo-2026 el crecimiento interanual real pasó de ~+4 % a ~+11 % (cambio de nivel). Con start 2026-05 o 2026-07 (meses previos observados) RF/XGB/Clima siguen subestimando 4-7 % (SARIMAX se adapta: 4.3 / 2.1 %). Con start 2026-01 nada puede anticiparlo.

## Idea (el agente valida/ajusta con evidencia)
Post-proceso por petición: con los meses observados antes de `start_date`, calcular residuos FUERA DE MUESTRA del propio modelo (pronóstico hecho desde un origen anterior a esos meses vs real) y estimar un factor de nivel multiplicativo con suavizado exponencial / media recortada de log(real/pron) de los últimos K meses, amortiguado (phi) y decreciendo con el horizonte (vida media h). Aplicar a los meses pronosticados. Sin meses observados previos (start = fin del histórico + 1 sin residuos) => factor 1 (sin cambio).
Hiperparámetros (K, phi, vida media, método) elegidos SOLO con orígenes ≤ 2023-12 (objetivos ≤ 2025-12); validar en 2024-2025; 2026 se mira una vez al final. Debe mejorar de forma robusta (VAL y 2026, no solo en uno) y NO empeorar start=2026-01/orígenes sin shock; límite al factor (p.ej. recorte a rango estadístico justificado, no hardcodeado a 2026).

## Reglas
- Cero fuga, walk-forward con origenes mensuales; residuos con el mismo proceso de inferencia de la API.
- Metodología/rigor igual que docs/work/09_clima_v3 (reporte honesto, comparar contra sin corrección, naive+drift y SARIMAX, por periodo y por horizonte/start).
- Contrato API intacto (campos, fechas, escalas); activar por defecto solo si gana; flag de rollback `LEVEL_ADAPT=0`. Exponer en respuesta un campo opcional `ajuste_nivel` (factor aplicado) para transparencia.
- Cap/override de forecast_adjustments no se tocan. Costo de latencia medido (cache existente).
- Tests nuevos, `python -m pytest tests -q` verde (hoy 281). Documentar en REPORTE.md y sessions/2026-11-23-mejora-modelos-forecast.md.

## Estado
- [ ] implementación  - [ ] revisión multi-ángulo + regresión  - [ ] correcciones  - [ ] commit
