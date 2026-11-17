"""Corrector de post-predicción para el pronóstico de demanda clima+ONI.

Módulo genérico y puro (sin I/O de base de datos) que replica, para el
problema de sesgo/estacionalidad de la demanda mensual/anual pronosticada,
el mismo patrón ya validado en ``scripts/calibrate_mpm_climate_adjusters.py``
y ``api/services/mpm_legacy/climate_adjuster.py`` para el ajuste climático
diario del MPM: un "prior" conservador (no corregir) mezclado con un ajuste
calibrado a datos recientes, con fallback al prior cuando hay pocas
observaciones, y exportación/carga de un artefacto JSON versionado.

Todas las funciones reciben ``pandas`` Series/valores como entrada y no
tocan el sistema de archivos salvo ``export_corrector_artifact`` y
``load_corrector_artifact``, que son los únicos puntos de I/O explícitos.

Este módulo es independiente del dataset/entrenamiento del modelo base de
pronóstico (``training/forecast_climate/build_dataset.py`` /
``features.py``): solo consume pares predicción/real ya alineados.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de calibración
# ---------------------------------------------------------------------------

# Mínimo de observaciones recientes para confiar en el ajuste calibrado del
# sesgo reciente; por debajo de esto se usa el prior puro (no corregir).
MIN_OBS = 3

# Mínimo de observaciones históricas por mes calendario para calibrar su
# corrector estacional; por debajo de esto ese mes queda sin corrección (0).
MIN_OBS_POR_MES_DEFAULT = 3

# Ventana por defecto (en meses) de historia reciente usada para el sesgo.
VENTANA_MESES_DEFAULT = 6

# Peso por defecto del prior (0% de corrección) al mezclar con el ajuste
# calibrado del sesgo reciente. Igual que en climate_adjuster.py, el prior
# domina cuando hay pocos datos o se prefiere conservadurismo.
PESO_PRIOR_DEFAULT = 0.7

ARTIFACT_VERSION = "v1"


# ---------------------------------------------------------------------------
# 1. Corrector de sesgo reciente
# ---------------------------------------------------------------------------


def calibrate_bias_corrector(
    predicciones: pd.Series,
    valores_reales: pd.Series,
    ventana_meses: int = VENTANA_MESES_DEFAULT,
    peso_prior: float = PESO_PRIOR_DEFAULT,
) -> dict[str, Any]:
    """Calibra un corrector de sesgo porcentual sobre el error reciente.

    ``predicciones`` y ``valores_reales`` deben estar alineados por índice
    (mismo orden temporal, típicamente índice de fecha/periodo creciente).
    Se toman los últimos ``ventana_meses`` pares disponibles (los más
    recientes según el orden de los Series, no una fecha explícita).

    El prior es 0% de corrección (confiar en el modelo crudo). El ajuste
    calibrado es el promedio del error porcentual relativo reciente:
    ``error_pct = (real - pred) / real``. La mezcla final es:

        correccion_final = (1 - peso_prior) * ajuste_calibrado + peso_prior * 0

    Si hay menos de ``MIN_OBS`` observaciones utilizables, se cae al prior
    puro (corrección 0) y ``peso_prior_usado`` queda en 1.0 para dejar
    constancia del fallback.

    Devuelve un dict con ``correccion_pct``, ``n_observaciones``,
    ``metodo`` y ``peso_prior_usado``.
    """

    pred = pd.Series(predicciones).reset_index(drop=True)
    real = pd.Series(valores_reales).reset_index(drop=True)
    if len(pred) != len(real):
        raise ValueError(
            "predicciones y valores_reales deben tener la misma longitud "
            f"(recibido {len(pred)} vs {len(real)})"
        )

    pares = pd.DataFrame({"pred": pred, "real": real}).dropna()
    # Evita división por cero: valores reales nulos o cero no aportan señal.
    pares = pares[pares["real"] != 0]

    ventana = pares.tail(max(int(ventana_meses), 0))
    n_obs = int(len(ventana))

    if n_obs < MIN_OBS:
        return {
            "correccion_pct": 0.0,
            "n_observaciones": n_obs,
            "metodo": "prior_puro_datos_insuficientes",
            "peso_prior_usado": 1.0,
        }

    error_pct = (ventana["real"] - ventana["pred"]) / ventana["real"]
    ajuste_calibrado = float(error_pct.mean())
    peso_prior = float(peso_prior)
    correccion_final = (1.0 - peso_prior) * ajuste_calibrado + peso_prior * 0.0

    return {
        "correccion_pct": correccion_final,
        "n_observaciones": n_obs,
        "metodo": "promedio_error_reciente_prior_blend",
        "peso_prior_usado": peso_prior,
    }


# ---------------------------------------------------------------------------
# 2. Corrector estacional residual
# ---------------------------------------------------------------------------


def calibrate_seasonal_corrector(
    predicciones: pd.Series,
    valores_reales: pd.Series,
    meses: pd.Series,
    min_obs_por_mes: int = MIN_OBS_POR_MES_DEFAULT,
) -> dict[str, Any]:
    """Calibra un corrector de residual estacional promedio por mes.

    Para cada mes calendario (1-12) calcula el residual porcentual promedio
    histórico ``(real - pred) / real`` usando todas las observaciones
    disponibles de ese mes. Si un mes tiene menos de ``min_obs_por_mes``
    observaciones, queda sin corrección (0.0) para evitar sobreajustar a
    pocos datos.

    Devuelve un dict con:
      - ``correcciones``: {mes (1-12): correccion_pct}
      - ``n_observaciones_por_mes``: {mes (1-12): n}
      - ``metodo``
      - ``min_obs_por_mes``
    """

    pred = pd.Series(predicciones).reset_index(drop=True)
    real = pd.Series(valores_reales).reset_index(drop=True)
    mes = pd.Series(meses).reset_index(drop=True)
    if not (len(pred) == len(real) == len(mes)):
        raise ValueError(
            "predicciones, valores_reales y meses deben tener la misma "
            f"longitud (recibido {len(pred)}, {len(real)}, {len(mes)})"
        )

    datos = pd.DataFrame({"pred": pred, "real": real, "mes": mes}).dropna()
    datos = datos[datos["real"] != 0]
    datos["mes"] = datos["mes"].astype(int)
    datos["error_pct"] = (datos["real"] - datos["pred"]) / datos["real"]

    correcciones: dict[int, float] = {}
    n_por_mes: dict[int, int] = {}
    for mes_num in range(1, 13):
        subset = datos[datos["mes"] == mes_num]
        n = int(len(subset))
        n_por_mes[mes_num] = n
        if n >= int(min_obs_por_mes):
            correcciones[mes_num] = float(subset["error_pct"].mean())
        else:
            correcciones[mes_num] = 0.0

    return {
        "correcciones": correcciones,
        "n_observaciones_por_mes": n_por_mes,
        "metodo": "residual_promedio_por_mes_calendario",
        "min_obs_por_mes": int(min_obs_por_mes),
    }


# ---------------------------------------------------------------------------
# 3. Aplicación de correctores
# ---------------------------------------------------------------------------


def apply_correctors(
    prediccion_cruda: float,
    mes: int,
    bias_corrector: dict[str, Any],
    seasonal_corrector: dict[str, Any],
) -> dict[str, Any]:
    """Aplica el corrector estacional y luego el de sesgo reciente.

    Orden deliberado: primero el corrector ESTACIONAL, porque corrige un
    patrón estructural repetitivo del calendario (p.ej. diciembre
    sistemáticamente sub-predicho); luego el corrector de SESGO RECIENTE,
    porque corrige la deriva más reciente del modelo *sobre* la predicción
    ya corregida estacionalmente (el sesgo reciente ya "ve" el efecto
    estacional en los datos con que se calibró, así que se aplica encima,
    no en paralelo, para no contar el mismo error dos veces de forma
    inconsistente).

    Ambos ajustes son multiplicativos: ``pred * (1 + correccion_pct)``.

    Devuelve un dict con cada etapa por separado para permitir auditoría:
      - ``prediccion_cruda``
      - ``prediccion_ajustada_estacional``
      - ``prediccion_final``
      - ``correcciones_aplicadas``: {"estacional": pct, "sesgo_reciente": pct}
    """

    pred_cruda = float(prediccion_cruda)
    mes_int = int(mes)

    # Las claves de mes pueden llegar como int (dict recién calibrado en
    # memoria) o como str (dict recargado desde JSON, donde las claves de
    # objeto siempre son cadenas) -> se prueban ambas representaciones.
    correcciones_mes = seasonal_corrector.get("correcciones", {})
    correccion_estacional = correcciones_mes.get(
        mes_int, correcciones_mes.get(str(mes_int), 0.0)
    )
    correccion_estacional = float(correccion_estacional)
    pred_estacional = pred_cruda * (1.0 + correccion_estacional)

    correccion_sesgo = float(bias_corrector.get("correccion_pct", 0.0))
    pred_final = pred_estacional * (1.0 + correccion_sesgo)

    return {
        "prediccion_cruda": pred_cruda,
        "prediccion_ajustada_estacional": pred_estacional,
        "prediccion_final": pred_final,
        "correcciones_aplicadas": {
            "estacional": correccion_estacional,
            "sesgo_reciente": correccion_sesgo,
        },
    }


# ---------------------------------------------------------------------------
# 4. Exportación de artefacto versionado
# ---------------------------------------------------------------------------


def export_corrector_artifact(
    bias_corrector: dict[str, Any],
    seasonal_corrector: dict[str, Any],
    output_path: str,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Exporta bias_corrector + seasonal_corrector a un JSON versionado.

    Sigue el espíritu de versionado de ``rn_baq_v1.json``: ``version``,
    metadata de trazabilidad (``train_meta``) y los correctores completos
    para que puedan auditarse o recalibrarse.
    """

    n_meses_total = sum(
        seasonal_corrector.get("n_observaciones_por_mes", {}).values()
    )
    train_meta: dict[str, Any] = {
        "n_meses_total": int(n_meses_total),
        "n_observaciones_bias": bias_corrector.get("n_observaciones"),
    }
    if extra_meta:
        train_meta.update(extra_meta)

    artifact = {
        "version": ARTIFACT_VERSION,
        "fecha_calibracion": datetime.now(timezone.utc).isoformat(),
        "bias_corrector": bias_corrector,
        "seasonal_corrector": seasonal_corrector,
        "metodo": "bias_reciente + estacional_residual",
        "train_meta": train_meta,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. Carga del artefacto para uso en producción
# ---------------------------------------------------------------------------


def load_corrector_artifact(path: str) -> dict[str, Any]:
    """Carga un artefacto de corrector exportado por ``export_corrector_artifact``."""

    artifact_path = Path(path)
    with artifact_path.open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Versión de artefacto no soportada en {artifact_path}: "
            f"{artifact.get('version')!r}"
        )
    return artifact
