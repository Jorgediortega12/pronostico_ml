"""Macro constante para SARIMAX (id=3), igual que el original (GMR Atenea).

En producción el original no usa las proyecciones de la BD para los años futuros:
sostiene constante el último año conocido de `api/data/gmr_annual_es.csv`
(mismas columnas macro que usa el modelo SARIMAX).
"""
from pathlib import Path

import pandas as pd

MACRO_COLUMNS = [
    "ipc",
    "Crecimiento de la población (%) anual",
    "Inflación, deflactor del PIB: series vinculadas (% anual)",
    "Comercio de mercaderías (% del PIB)",
    "Importaciones de bienes (balanza de pagos, US$ a precios actuales)",
    "Empleo de tiempo parcial, total (% del total de empleo)",
]

_MACRO_ANNUAL_CSV = Path(__file__).resolve().parents[1] / "data" / "gmr_annual_es.csv"


def build_future_macro_vars(n_years: int, start_year: int = None) -> pd.DataFrame:
    """DataFrame indexado por año (uno por año a pronosticar) con el último año conocido repetido."""
    hist_annual = pd.read_csv(_MACRO_ANNUAL_CSV).sort_values("año").set_index("año")
    last_row = hist_annual.iloc[-1]
    if start_year is None:
        start_year = int(hist_annual.index.max()) + 1
    future_rows = [{col: last_row[col] for col in MACRO_COLUMNS} for _ in range(n_years)]
    idx = pd.RangeIndex(start=start_year, stop=start_year + n_years)
    return pd.DataFrame(future_rows, index=idx, columns=MACRO_COLUMNS)
