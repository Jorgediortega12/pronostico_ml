# REPORTE agente-correcciones (ronda 1 -> 2)
252 tests pasan (2 corridas). Contrato HTTP intacto; único campo nuevo opcional: `modelo.meses_hueco_rellenados`.
- A1 (alto, honestidad): README y REPORTEs reportan el corte post-selección (2024-01..2025-12): RF 6.39, XGB 6.11, naive+drift 4.67; Climático v2 3.97 (sin ONI 3.91); SARIMAX 4.19 vs 4.24. Sin mezcla con naive+drift. Test 2026 = 1 origen, 8 meses.
- IPC a mitad de año: solo escala meses pronosticados (`apply_ipc_adjustment(fixed_by_year, adjust_from)`).
- Tope post-división: actúa solo si factor_division == 1.0; con 70% no actúa (sep/oct ~ -5..-9% visibles). Con BD local 100% actúa (error crudo +28..+35% queda en `errores_porcentuales_crudos`).
- Rendimiento: `forecast_cache.py` (LRU real, single-flight, límite 2 reentrenos); repetida 2.8s -> 0.4-0.6s; RF/XGB n_jobs=2.
- Climático: n_years_shape=2 (MAPE servido ene-ago 2026 sube 4.45 -> 5.07); growth con año previo sin reales; start>=2028 -> ValueError claro; pérdida de sensibilidad ENSO (9.7%->3.1%) aceptada.
- Hueco interno rellenado con pronóstico; RF/XGB historia mínima 48 meses con error claro; get_number_of_days usa calendar_counts (ene-2026: 20/5/6); logging; CSV de detalle en gzip; DivisionMercado hereda de MarketBreak; read_raw_daily_session delega en read_raw_daily.
- ML_FORECAST_LEGACY documentado como rollback parcial (solo inferencia).
- Aceptado: corrector_ultimo_origen.pkl se conserva (lo lee train_final_clima.py).
