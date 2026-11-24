# Sub-plan agente-sarimax (id=3)

Lee: `docs/work/PLAN.md`, `docs/work/00_entorno/DIAGNOSTICO.md` §1/§4-id3, `docs/work/01_datos/REPORTE.md`.

## Problemas del original
1. Ignora `start_date` (siempre pronostica desde 2026-01 y rotula desde el último dato).
2. Orden (5,1,6)(1,1,1,12) sobreparametrizado; MAPE 1.6 % reportado es in-sample.
3. Endog 2020-Q4 inflado ~10 % (duplicados) y COVID sin tratar.
4. Exógenas macro anuales sostenidas constantes a futuro (no aportan información en inferencia).

## Protocolo (cero fuga)
- Datos: `training.common.data.load_monthly_demand(impute_incomplete=True)`; train = `target_series(.., market_break=mb)`
  (corte en el quiebre de mercado 2026-09); reales = sin imputados.
- **Selección** (orden, transformación, COVID, exógenas, ensamble): walk-forward expanding con orígenes mensuales
  2017-12..2023-12, h=24, reales truncados a <= 2025-12 (2026 nunca se mira), métricas sin COVID. AIC/BIC con train <= 2023-12
  como criterio secundario / filtro de órdenes.
- **Reporte dev**: walk-forward orígenes 2017-12..2025-12 con reales <= 2025-12 (antes/después vs original vs naive+drift).
- **Test final 2026** (una sola vez, al final): origen 2025-12, h=1..8 (ene-ago 2026), modelo elegido sin re-tunear.
- Original reevaluado honestamente: mismo orden/macro constante, reajustado en cada origen (walk-forward), además del
  artefacto tal cual para 2026.

## Candidatos
- SARIMA parsimoniosos sobre log(y) y log(y/días-del-mes) — rejilla p,q∈{0,1,2}, P,Q∈{0,1}, d=1, D=1 (y d=0,D=1 con
  tendencia), filtrados por AIC en train<=2023-12.
- COVID: (a) nada, (b) meses 2020-03..2021-02 como NaN (Kalman los salta), (c) dummies de intervención.
- Exógenas de calendario deterministas (disponibles siempre en inferencia): días hábiles / festivos (holidays_co).
  Macro descartada salvo evidencia (en inferencia solo habría constantes).
- ETS/Holt-Winters amortiguado sobre log; ensamble simple (media) si gana en validación.

## Entregables
- `training/forecast_sarimax/` (`candidates.py`, `select.py`, `train_final.py`, `final_test.py`).
- `api/services/forecast_sarimax.py`: `predict_monthly(history, start, end, ...)` autocontenido; re-ancla con
  `res.apply` sobre reales < start; re-estima parámetros si el artefacto vio datos >= start (`refit="auto"`).
- `api/models/sarimax_model/` nuevos artefactos (originales ya respaldados en `docs/work/backup_modelos_originales/models`).
- `tests/forecast_sarimax/`.
- `REPORTE.md` con tabla antes/después y test 2026.

## Progreso
- [x] Lectura de contexto y diseño.
- [x] Etapa 1 AIC/BIC (`s1_aic.csv`): top BIC = airline (0,1,1)(0,1,1) y d=0+constante.
- [x] Etapa 2 (`select_s2_*`): gana familia (2,0,0)(0,1,1,12)+c, COVID sin tratar, 4.10 % vs naive+drift 6.57 %
  vs original reajustado 7.60 % (mismos orígenes 2017-12..2023-12); mejor en todos los años de origen.
- [x] Etapa 2b/3/ens: elegido (1,0,0)(0,1,1,12)+c log_daily, covid none, sin exógenas (4.04 %).
- [x] Módulo inferencia + artefacto `sarimax_v2.json`.
- [x] Tests (19) + pytest completo.
- [x] Test final 2026 + REPORTE.md.
