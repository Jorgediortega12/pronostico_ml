# REPORTE correcciones ronda 2
266 tests pasan (2 corridas); pyflakes sin avisos nuevos en lo tocado.
1. Fechas: ids 1-3 fin de mes (como main), id 4 inicio de mes; `contract_month_labels`; verificado contra main en worktree temporal.
2. Caché reentrenos RF/XGB 8->32. 3. Historia mínima = `min_training_months` (58 meses; ids 1-2 desde ~2015-11, id 3 desde 2014-01) con mensaje claro.
4. Honestidad: solo objetivos >=2026 (n=164) son OOS puro (RF 5.92, XGB 5.79, SARIMAX 4.03, naive+drift 5.40, Clima 3.83); `nota_metricas` en metadata; id 4 no reentrena en hindcast; no había mojibake (era la consola).
5. CSV e2e regenerados (Clima 5.07 crudo / 4.01 con tope). 6. `api/services/calendar_co.py` único (sin fallback silencioso; salida idéntica 2010-2030).
7. Simplificaciones: clave SARIMAX sin start; _error_metrics (JSON idéntico); helpers sin uso borrados; test de humo ML_FORECAST_LEGACY=1.
8. README/documentación de contrato y reversión (`git checkout main -- api/models/...`).
Aceptado: analysis.py conteo de días propio (preexistente); features.py cita calibrate_oni_decay.py (archivo sin cambios de main).
