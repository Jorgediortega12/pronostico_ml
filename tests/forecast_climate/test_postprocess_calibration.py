"""Tests del corrector post-predicción para el pronóstico clima+ONI.

Usa datos sintéticos (no depende del dataset real que construye otro
agente en paralelo en training/forecast_climate/build_dataset.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from training.forecast_climate.postprocess_calibration import (  # noqa: E402
    MIN_OBS,
    apply_correctors,
    calibrate_bias_corrector,
    calibrate_seasonal_corrector,
    export_corrector_artifact,
    load_corrector_artifact,
)


def _synthetic_series(n_meses: int = 36, sesgo_pct: float = 0.0, seed: int = 42):
    """Genera predicción/real/mes sintéticos con sesgo y estacionalidad.

    - Demanda base con tendencia leve.
    - Estacionalidad: diciembre (mes 12) tiene un salto real no capturado
      por el modelo (sub-predicción sistemática) para poder probar el
      corrector estacional.
    - ``sesgo_pct``: sub-predicción sistemática adicional aplicada a TODA
      la serie (para probar el corrector de sesgo reciente).
    """

    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2022-01-01", periods=n_meses, freq="MS")
    meses = pd.Series(fechas.month, name="mes")

    base = 100.0 + np.arange(n_meses) * 0.3
    ruido = rng.normal(0, 0.5, size=n_meses)
    real = pd.Series(base + ruido, name="real")

    # Diciembre: la demanda real es sistemáticamente ~8% más alta que lo
    # que "predeciría" un modelo ciego a esa estacionalidad.
    es_diciembre = meses == 12
    real_con_estacionalidad = real.copy()
    real_con_estacionalidad[es_diciembre] = real[es_diciembre] * 1.08

    # La predicción cruda del modelo no conoce el salto de diciembre y,
    # opcionalmente, tiene un sesgo sistemático adicional (sub-predicción).
    pred = real.copy()
    if sesgo_pct:
        pred = pred * (1.0 - sesgo_pct)

    return pred.reset_index(drop=True), real_con_estacionalidad.reset_index(drop=True), meses.reset_index(drop=True)


class TestBiasCorrector:
    def test_pocos_datos_cae_a_prior(self):
        pred = pd.Series([100.0, 101.0])
        real = pd.Series([100.0, 101.0])
        resultado = calibrate_bias_corrector(pred, real, ventana_meses=6)

        assert resultado["n_observaciones"] < MIN_OBS
        assert resultado["correccion_pct"] == pytest.approx(0.0)
        assert resultado["metodo"] == "prior_puro_datos_insuficientes"
        assert resultado["peso_prior_usado"] == 1.0

    def test_sesgo_sistematico_se_detecta_y_corrige(self):
        # Predicciones consistentemente 5% por debajo de lo real.
        pred, real, _ = _synthetic_series(n_meses=24, sesgo_pct=0.05)
        resultado = calibrate_bias_corrector(
            pred, real, ventana_meses=12, peso_prior=0.5
        )

        assert resultado["n_observaciones"] == 12
        # error_pct = (real - pred) / real ~ 0.05 (pred está 5% por debajo)
        # con peso_prior=0.5, la corrección final se regulariza a la mitad.
        assert 0.0 < resultado["correccion_pct"] < 0.05
        assert resultado["correccion_pct"] == pytest.approx(0.05 * 0.5, abs=0.01)
        assert resultado["metodo"] == "promedio_error_reciente_prior_blend"

    def test_ventana_toma_los_mas_recientes(self):
        # Sesgo solo en los últimos meses; con ventana chica debe detectarse
        # con más fuerza que si se promediara toda la serie.
        pred_sin_sesgo, real, _ = _synthetic_series(n_meses=24, sesgo_pct=0.0)
        pred_con_sesgo_reciente = pred_sin_sesgo.copy()
        pred_con_sesgo_reciente.iloc[-6:] = real.iloc[-6:] * 0.90  # -10% reciente

        resultado_ventana_chica = calibrate_bias_corrector(
            pred_con_sesgo_reciente, real, ventana_meses=6, peso_prior=0.0
        )
        resultado_ventana_grande = calibrate_bias_corrector(
            pred_con_sesgo_reciente, real, ventana_meses=24, peso_prior=0.0
        )

        assert resultado_ventana_chica["correccion_pct"] > resultado_ventana_grande["correccion_pct"]


class TestSeasonalCorrector:
    def test_diciembre_subpredicho_detectado(self):
        pred, real, meses = _synthetic_series(n_meses=36, sesgo_pct=0.0)
        resultado = calibrate_seasonal_corrector(pred, real, meses, min_obs_por_mes=2)

        # 36 meses -> 3 diciembres, suficientes para calibrar (>=2).
        assert resultado["n_observaciones_por_mes"][12] >= 2
        assert resultado["correcciones"][12] > 0.03  # sube ~8%, con margen

        # Meses "normales" (sin el salto simulado) deben quedar cerca de 0.
        assert abs(resultado["correcciones"][6]) < 0.02

    def test_mes_con_pocos_datos_queda_sin_correccion(self):
        pred, real, meses = _synthetic_series(n_meses=36, sesgo_pct=0.0)
        # min_obs_por_mes muy alto -> ningún mes alcanza el mínimo.
        resultado = calibrate_seasonal_corrector(
            pred, real, meses, min_obs_por_mes=100
        )

        for mes_num in range(1, 13):
            assert resultado["correcciones"][mes_num] == 0.0

    def test_devuelve_los_doce_meses(self):
        pred, real, meses = _synthetic_series(n_meses=36)
        resultado = calibrate_seasonal_corrector(pred, real, meses)

        assert set(resultado["correcciones"].keys()) == set(range(1, 13))
        assert set(resultado["n_observaciones_por_mes"].keys()) == set(range(1, 13))


class TestApplyCorrectors:
    def test_orden_estacional_luego_sesgo(self):
        bias_corrector = {"correccion_pct": 0.10}
        seasonal_corrector = {"correcciones": {12: 0.08}}

        resultado = apply_correctors(
            prediccion_cruda=100.0,
            mes=12,
            bias_corrector=bias_corrector,
            seasonal_corrector=seasonal_corrector,
        )

        esperado_estacional = 100.0 * 1.08
        esperado_final = esperado_estacional * 1.10

        assert resultado["prediccion_cruda"] == pytest.approx(100.0)
        assert resultado["prediccion_ajustada_estacional"] == pytest.approx(esperado_estacional)
        assert resultado["prediccion_final"] == pytest.approx(esperado_final)
        assert resultado["correcciones_aplicadas"] == {
            "estacional": pytest.approx(0.08),
            "sesgo_reciente": pytest.approx(0.10),
        }

    def test_mes_sin_correccion_estacional(self):
        bias_corrector = {"correccion_pct": 0.05}
        seasonal_corrector = {"correcciones": {1: 0.0}}

        resultado = apply_correctors(
            prediccion_cruda=200.0,
            mes=1,
            bias_corrector=bias_corrector,
            seasonal_corrector=seasonal_corrector,
        )

        assert resultado["prediccion_ajustada_estacional"] == pytest.approx(200.0)
        assert resultado["prediccion_final"] == pytest.approx(200.0 * 1.05)

    def test_funciona_con_claves_de_mes_como_string(self):
        # Simula un seasonal_corrector recargado desde JSON (claves str).
        bias_corrector = {"correccion_pct": 0.0}
        seasonal_corrector = {"correcciones": {"12": 0.08}}

        resultado = apply_correctors(
            prediccion_cruda=100.0,
            mes=12,
            bias_corrector=bias_corrector,
            seasonal_corrector=seasonal_corrector,
        )

        assert resultado["prediccion_final"] == pytest.approx(108.0)


class TestArtifactRoundtrip:
    def test_export_load_roundtrip(self, tmp_path):
        pred, real, meses = _synthetic_series(n_meses=36, sesgo_pct=0.05)
        bias_corrector = calibrate_bias_corrector(pred, real, ventana_meses=12)
        seasonal_corrector = calibrate_seasonal_corrector(pred, real, meses, min_obs_por_mes=2)

        output_path = tmp_path / "postproc_corrector_demo.json"
        export_corrector_artifact(
            bias_corrector,
            seasonal_corrector,
            str(output_path),
            extra_meta={"source": "synthetic_test"},
        )

        assert output_path.is_file()

        loaded = load_corrector_artifact(str(output_path))

        assert loaded["version"] == "v1"
        assert "fecha_calibracion" in loaded
        assert loaded["metodo"] == "bias_reciente + estacional_residual"
        assert loaded["bias_corrector"]["correccion_pct"] == pytest.approx(
            bias_corrector["correccion_pct"]
        )
        assert loaded["bias_corrector"]["n_observaciones"] == bias_corrector["n_observaciones"]

        # Las claves de mes vuelven como string tras el roundtrip por JSON;
        # se comparan por valor normalizando a int.
        loaded_correcciones = {
            int(k): v for k, v in loaded["seasonal_corrector"]["correcciones"].items()
        }
        assert loaded_correcciones == pytest.approx(seasonal_corrector["correcciones"])

        assert loaded["train_meta"]["source"] == "synthetic_test"
        assert loaded["train_meta"]["n_meses_total"] == sum(
            seasonal_corrector["n_observaciones_por_mes"].values()
        )

    def test_artefacto_cargado_es_usable_en_apply_correctors(self, tmp_path):
        pred, real, meses = _synthetic_series(n_meses=36, sesgo_pct=0.05)
        bias_corrector = calibrate_bias_corrector(pred, real, ventana_meses=12)
        seasonal_corrector = calibrate_seasonal_corrector(pred, real, meses, min_obs_por_mes=2)

        output_path = tmp_path / "postproc_corrector_demo.json"
        export_corrector_artifact(bias_corrector, seasonal_corrector, str(output_path))
        loaded = load_corrector_artifact(str(output_path))

        resultado = apply_correctors(
            prediccion_cruda=100.0,
            mes=12,
            bias_corrector=loaded["bias_corrector"],
            seasonal_corrector=loaded["seasonal_corrector"],
        )

        assert resultado["prediccion_final"] > 0
        assert resultado["correcciones_aplicadas"]["estacional"] == pytest.approx(
            seasonal_corrector["correcciones"][12]
        )
