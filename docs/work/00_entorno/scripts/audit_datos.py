"""Auditoria de la demanda mensual en PRONOSTICO_demands (BD de .env / ML_DATABASE_URL).

Imprime por mes: file_type, filas, dias distintos vs dias del mes, GWh (suma de `total`), nulos,
filas donde total != suma(hora_1..24), rango diario, dias duplicados y crecimiento interanual.
Guarda docs/work/00_entorno/audit_demanda_mensual.csv.

Uso: python docs/work/00_entorno/scripts/audit_datos.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
url = os.getenv("ML_DATABASE_URL", "postgresql+psycopg2://postgres@127.0.0.1:55432/afinia")
eng = create_engine(url)

hsum = "+".join(f"coalesce(hora_{i},0)" for i in range(1, 25))
q = f'''
SELECT date_trunc('month', fecha)::date AS mes, file_type,
       COUNT(*) AS filas, COUNT(DISTINCT fecha::date) AS dias,
       SUM(total) AS total_kwh,
       SUM(CASE WHEN total IS NULL THEN 1 ELSE 0 END) AS nulos,
       SUM(CASE WHEN abs(total - ({hsum})) > 1 THEN 1 ELSE 0 END) AS total_vs_horas,
       MIN(total) AS min_dia, MAX(total) AS max_dia,
       string_agg(DISTINCT tipo_fecha, ',') AS tipos_fecha
FROM "PRONOSTICO_demands" GROUP BY 1, 2 ORDER BY 1, 2'''
with eng.connect() as c:
    d = pd.read_sql(q, c)
d["dias_mes"] = pd.to_datetime(d["mes"]).dt.days_in_month
d["GWh"] = d["total_kwh"] / 1e6
d["duplicados"] = d["filas"] - d["dias"]
d["incompleto"] = d["dias"] < d["dias_mes"]
txf = d[d.file_type == "txf"].set_index(pd.to_datetime(d.loc[d.file_type == "txf", "mes"]))
d.loc[d.file_type == "txf", "yoy_%"] = (txf["GWh"] / txf["GWh"].shift(12) - 1).values * 100
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 500)
print(d[["mes", "file_type", "filas", "dias", "dias_mes", "GWh", "yoy_%", "duplicados", "incompleto",
         "nulos", "total_vs_horas", "tipos_fecha"]].round(2).to_string())
print("\nAlertas:")
print(d[(d.duplicados > 0) | d.incompleto | (d.file_type != "txf")][["mes", "file_type", "filas", "dias", "dias_mes", "GWh"]].to_string())
out = ROOT / "docs" / "work" / "00_entorno" / "audit_demanda_mensual.csv"
d.to_csv(out, index=False)
print(f"\nGuardado {out}")
