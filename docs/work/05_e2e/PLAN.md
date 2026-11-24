# Sub-plan agente-e2e (05) — integración API de los modelos v2 + tope coherente

Lee: `docs/work/PLAN.md`, `00_entorno/DIAGNOSTICO.md`, reportes 01-04, `docs/work/04_clima/forecast_clima_patch.md`.

## Decisiones
1. **Historia limpia por petición**: nuevo `api/services/forecast_history.py` lee `PRONOSTICO_demands` con la
   MISMA sesión SQLAlchemy de la petición (`db.connection()`, nunca un engine global) y reutiliza las funciones
   puras de `training.common.data` (dedupe 2020-Q4, imputación 2021-09, meses completos/definitivos). `training/`
   se copia en la imagen Docker (`COPY ./training /api/training`), así que importarlo desde `api/` es válido.
   El quiebre de mercado se toma de `DemandConfigService.get_config(db)` (misma fuente/semántica que el front).
2. **Ruteo**: id 1/2 → `forecast_rfxgb.predict_monthly`; id 3 → `forecast_sarimax.predict_monthly`
   (`exclude_from` = primer mes post-quiebre); id 4 → código actual + artefactos v2 (el parche `n_years_shape`=5 se aplicó y luego se
   revirtió a 2 en 06_correcciones). Pronóstico SIEMPRE en escala de mercado completo (contrato de `predict_excel`: el
   consumidor multiplica todo el pronóstico por `porcentaje/100`). Rollback: `ML_FORECAST_LEGACY=1` usa el
   camino anterior para id 1/2/3.
3. **Años**: el pronóstico se calcula desde el último mes real usable hasta dic del último año pedido. Totales
   anuales = reales (escala completa) + pronóstico de los meses faltantes; los años previos a `start` con meses
   faltantes (p. ej. 2026 cuando start=2027-01) se completan con pronóstico solo para `growth`.
4. **Tope** (`cap_xgb_sarimax_error`, solo id 1/2, se mantiene): solo actúa en meses con real y **pre-división**
   (periodo que termina <= `fecha_limite`), donde real y pronóstico están ambos en escala de mercado completo, igual
   que los compara el front (ambos × factor). En meses post-división NUNCA actúa (real en otra escala; el
   porcentaje configurado puede estar desactualizado). [06_correcciones: excepción — si `factor == 1.0` la escala
   no cambia y el tope también actúa en meses post-división.] Errores reportados en la escala del front:
   pre: (pred−real)/real; post: (pred·f − real)/real.
5. **Override feb-2026**: se mantiene, pero solo aplica si el real de esa petición coincide (±0.5 %) con el valor
   del override (evita imponer el valor de Afinia en otra base) y si el mes tiene real.
6. **Campos opcionales nuevos** (no se quita ni renombra nada): por mes `valores_pronostico_crudos`,
   `errores_porcentuales_crudos`, `tope_aplicado`, `post_division`, `factor_division`; global
   `escala_pronostico`, `division_mercado`, `metricas_error_crudas`, `modelo`.
7. **Robustez**: periodos pedidos con `period_range` (rangos de 1 mes / días intermedios), JSON sin NaN/inf,
   sin requerir macro para modelos v2 (no la usan) → 2028 funciona.

## Pasos
- [x] forecast_history.py + tests
- [x] forecast.py ruteo v2 + post-proceso
- [x] forecast_adjustments.py (máscara de elegibilidad, guard override) + tests
- [x] tests/forecast_e2e (BD local, skip si no hay)
- [x] smoke HTTP uvicorn :8000 4 modelos × rangos → CSV en 05_e2e/
- [x] README + REPORTE
