"""Constantes de dominio de la capa común de datos/validación.

Se centralizan aquí (y no dentro de las funciones) para que sean explícitas, documentadas y
sobreescribibles por parámetro. Ninguna es un valor de demanda: son reglas de calidad/evaluación.
"""
from __future__ import annotations

import os

#: URL por defecto si no hay ``ML_DATABASE_URL`` en el entorno/.env (Postgres local del trabajo nocturno).
DEFAULT_DATABASE_URL = "postgresql+psycopg2://postgres@127.0.0.1:55432/afinia"

#: Tabla de demanda diaria (una fila por día, columnas hora_1..hora_24 y total en kWh).
DEMANDS_TABLE = "PRONOSTICO_demands"

#: Tabla de configuración de la división de mercado (la escribe el backend Node).
SPLIT_TABLE = "demand_split_config"

#: Tipo de archivo de datos definitivos. Cualquier otro (p. ej. ``tx2``) se trata como preliminar.
FINAL_FILE_TYPE = "txf"

#: Semiventana (días) para la referencia de escala al deduplicar fechas repetidas.
DEDUP_WINDOW_DAYS = 45

#: Tolerancia (kWh) para dar por consistente total vs Σ hora_1..24.
TOTAL_VS_HOURS_TOL_KWH = 1.0

#: Periodo COVID excluido de las MÉTRICAS (no del entrenamiento). Inclusivo, formato YYYY-MM.
COVID_EXCLUDED = ("2020-03", "2021-02")

#: Años de historia usados por el baseline naive estacional + drift.
DRIFT_YEARS = 3

#: Tramos de horizonte para reportar (meses, inclusivos).
HORIZON_BUCKETS = ((1, 12), (13, 24))


def database_url() -> str:
    """Devuelve ``ML_DATABASE_URL`` (cargando ``.env`` del repo si existe) o la URL local por defecto."""
    try:
        from pathlib import Path

        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:  # python-dotenv es opcional
        pass
    return os.getenv("ML_DATABASE_URL", DEFAULT_DATABASE_URL)
