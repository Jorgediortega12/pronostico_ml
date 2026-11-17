# SPEC — MPM legacy engine (surrogate calibrado)

## Objetivo
Reimplementar en Python el núcleo del fragmento MATLAB en
`references/npm_legacy.md` como motor puro (sin FastAPI/BD).

## Paridad
Estado: **calibrated-surrogate** (no `matlab-rn`).

- Completo: TipoDia, Prom_Dia (medianas), barrido de días similares
  (`for i = 1:3 / switch i`), lineal BAQ `dD = 1.099*dT - 0.01963`,
  AssembledData 10xn, PronosticoTotal `[f3, f5, f6, Prom_Dia·conteos]/1e6`.
- Adaptativo: f5 y f6 usan correctores residuales calibrados con datos
  históricos; no reproducen las redes MATLAB ausentes.
- Parcial: relleno climático por perfiles sintéticos (no `ClimaRef.mat`).

## Contrato del corrector climático
`ClimateAdjuster` opera exclusivamente sobre diferencias de temperatura y
devuelve diferencias de demanda en GW:

- `adjust_baq(dT) -> dD`
- `adjust_3c(dT_baq, dT_car, dT_sm) -> dD`

`CalibratedCurveAdjuster` carga los artefactos versionados
`artifacts/rn_baq_v1.json` y `artifacts/rn_3c_v1.json`. BAQ es una curva
piecewise-linear monótona; 3C es una regresión residual con coeficientes no
negativos. Cada entrada se recorta a `[-5, 5] °C` y la salida a `[-3, 3] GW`.
`FallbackChainAdjuster` usa el lineal legacy
`1.099*dT - 0.01963` si falta el artefacto o su salida no es finita/válida.
`MPM_CLIMATE_ADJUSTER=linear|calibrated` permite fijar el modo; por defecto
se intenta `calibrated`.

**Clipping asimétrico, a propósito.** `CalibratedCurveAdjuster` recorta
entrada y salida; `LinearFallbackAdjuster` **no** recorta ninguna de las dos:
evalúa la recta cruda para conservar la paridad exacta con `f4` del fragmento
MATLAB, que tampoco satura. Por eso, cuando la cadena cae al fallback, el
ajuste devuelto puede salirse de `[-3, 3] GW`.

Los artefactos se generaron offline con
`scripts/calibrate_mpm_climate_adjusters.py` desde
`Combined_Demanda_Clima.csv`. El calibrador empareja días ordinarios del mismo
weekday con referencias de las nueve semanas previas y temperaturas
auxiliares cercanas; excluye sábados, domingos y festivos colombianos
(`holidays_co`), y a igual cercanía climática desempata por la referencia más
reciente. El target es `(P_day-P_ref)/1e6`. BAQ usa isotonic con prior lineal;
3C usa ridge no negativo hacia ese mismo prior.

## Barrido de días similares (`similar_days.py`)
`AssembledData(3,:)` **no** se rellena día a día con la temperatura de cada
día futuro: se hace un barrido por tipo que escribe el mismo valor en todas
las columnas que comparten `(weekday, TipoDia)`.

| Case | Pool | Anclaje `PairDays` | Columnas escritas |
|------|------|--------------------|-------------------|
| 1 Ordinarios | `TipoDia == 0` | primer día del mes con `weekday ∈ 2:6` y tipo 0 | todas con ese `(weekday, tipo)` |
| 2 Sábados | `TipoDia == 1` | primer día del mes con `weekday == 7` (sin exigir tipo) | todas con ese `(7, tipo del anclaje)` |
| 3 Dom/festivos | `TipoDia == -1` | — (sin temperatura) | todas con `tipo == -1`, valor = media del pool |

Criterio de desviación, idéntico en case 1 y 2:

- `Desv = mean(DemandaActual) - Demanda(similar)`; sin día similar, `Desv = 8e5`.
- `Desv > 7e5` → media del mes actual (o media del pool si el mes actual está vacío).
- En otro caso → demanda del día similar.
- Si hay día similar pero el mes actual está vacío, `mean([])` es `NaN` y
  `NaN > 7e5` es falso: gana el día similar (no el baseline).

Case 2 sustituye `DemandaActual = Demanda` cuando no hay sábados del mes
actual, así que su baseline nunca cae al pool por la rama de vacío.

Al final, la demanda real observada pisa sus columnas, igual que
`AssembledData(3,1:size(DemandaActual,1)) = DemandaActual(:,2)'`.

## Fallback climático (`find_climate_reference`)
`pos = f - ProData(i,1)` usa `f` **1-based** sobre `RefData`, así que la
distancia se calcula como `(idx + 1) - target_weekday`. Prioridad:
`pos == -1`, luego `pos == 1`, luego `min|pos|`; siempre se toma la última
fila candidata (`f = f(end)`).

## Mapa de ciudades (provisional)
| Legacy | Serie app        |
|--------|------------------|
| Baq    | bolivar          |
| Car    | bolivar (misma)  |
| Mon    | cordoba_sucre    |
| SM     | cesar            |

## method_id → legacy
| id | Fuente                         |
|----|--------------------------------|
| 1  | sum(f3) días similares         |
| 2  | sum(f5) surrogate calibrado BAQ |
| 3  | sum(f6) surrogate calibrado 3C  |
| 4  | Prom_Dia × [NLab,NSáb,NFestDom]|

f4 (lineal) se calcula pero **no** entra en los 4 totales.

## Limitaciones conocidas
Diferencias aceptadas frente al fragmento MATLAB, por falta de datos o de
alcance del motor puro:

1. **Redes neuronales de ajuste climático.** No existen
   `RN_AjusteClimaBAQ.mat` ni `RN_AjusteClima3C.mat`. f5/f6 son surrogates
   residuales calibrados y conservan monotonía/sensibilidad, pero no prometen
   igualdad numérica con las RN MATLAB.
2. **`ClimaRef.mat` / perfiles sintéticos.** El relleno de días sin clima usa
   perfiles deterministas por escenario y día de semana (`climate_fill.py`)
   en lugar de las series de referencia Caluroso/Soleado/Fresco del legacy.
3. **Baq ≡ Car.** La app solo expone tres series climáticas
   (`bolivar`, `cordoba_sucre`, `cesar`), de modo que Barranquilla y
   Cartagena comparten `bolivar`. En calibración, `t_CTG` se duplica para
   Baq/Car y `t_VDP` representa SM; `t_MON` solo restringe el matching. Por
   tanto el modelo 3C degenera de forma explícita a dos señales independientes.
4. **Exclusión de días atípicos (columna 107).** El legacy hace
   `DataBaseED(DataBaseED(:,107) == 0,:) = []` antes de calcular. Aquí el
   filtro vive en la capa de datos: el motor recibe `demand_by_date` y
   `previous_days_demand` ya depurados.
5. **Días previos sin temperatura.** `previous_days_demand` admite dos
   contratos: `(fecha, demanda)` y `(fecha, demanda, temp_baq)`. Con la
   3-tupla el día previo entra al match por temperatura de `RefDays` igual que
   en MATLAB. Con la 2-tupla se marca `temp_baq = NaN`: alimenta las medias de
   los pools pero queda fuera del match, porque `round_temp` propaga NaN y
   `NaN != NaN`. La brecha es solo de disponibilidad de datos en la capa que
   llama al motor, no del motor.
6. **Pools vacíos.** MATLAB devuelve `NaN` al promediar un pool vacío; aquí se
   devuelve `0.0` para no propagar NaN a `PronosticoTotal`.
7. **Observados fuera del prefijo.** El legacy asume que los días observados
   son el prefijo `1..N` del mes. La versión Python pisa con demanda real
   cualquier día presente en `demand_by_date`, aunque haya huecos; el ajuste
   climático sí sigue usando el conteo de prefijo.
