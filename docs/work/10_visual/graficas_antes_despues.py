"""Graficas por modelo: real vs baseline (antes de la mejora, docs/work/00_entorno/baseline_2026_2027.csv) vs ahora (API real).
Uso: python docs/work/10_visual/graficas_antes_despues.py   -> PNG en docs/work/10_visual/"""
import importlib.util
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]; OUT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("bl", ROOT/"docs/work/00_entorno/scripts/baseline.py")
bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
db = bl.SessionLocal()
old = pd.read_csv(ROOT/"docs/work/00_entorno/baseline_2026_2027.csv", index_col="mes"); old.index = pd.PeriodIndex(old.index, freq="M")
old.loc[old.index > pd.Period("2026-08","M"), "real"] = float("nan")  # desde 2026-09 el real esta al 70 % del mercado: no comparable
names = {1: ("RF", "Random Forest"), 2: ("XGB", "XGBoost"), 3: ("SARIMAX", "SARIMAX"), 4: ("CLIMA_ONI", "Climático + ONI")}
fig_all, axs = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
rows = []
for ax, (mid, (k, title)) in zip(axs.ravel(), names.items()):
    now, _ = bl.run_model(db, mid, "2026-01-01", "2027-12-31", raw=True)
    for a in (ax, None):
        fig = None
        if a is None:
            fig, a = plt.subplots(figsize=(11, 5.5))
        x = old.index.to_timestamp()
        a.plot(x, old["real"]/1e6, "k-o", ms=4, lw=2, label="Real (ene-ago 2026)")
        a.plot(x, old[f"{k}_crudo"]/1e6, "--", color="#d62728", lw=1.8, label="Antes (baseline de ayer)")
        a.plot(x, now.reindex(old.index)/1e6, "-", color="#2ca02c", lw=2.2, label="Ahora (mejorado)")
        a.plot(x, old["NAIVE_EST_DRIFT"]/1e6, ":", color="gray", lw=1.3, label="Naive+drift (referencia)")
        a.axvline(pd.Timestamp("2026-08-31"), color="#999", lw=.8); a.text(pd.Timestamp("2026-09-10"), a.get_ylim()[0], "real no comparable (división de mercado)", fontsize=7, color="#777", va="bottom")
        a.set_title(title); a.set_ylabel("GWh/mes"); a.grid(alpha=.25); a.legend(fontsize=8, loc="upper left")
        if fig is not None:
            fig.suptitle(f"{title}: pronóstico crudo 2026-2027 vs baseline anterior (escala 100 % mercado)")
            fig.tight_layout(); fig.savefig(OUT/f"modelo_{mid}_{k.lower()}.png", dpi=110); plt.close(fig)
    m = old["real"].notna()
    e_old = ((old[f"{k}_crudo"]-old["real"])/old["real"]*100)[m].abs().mean()
    e_new = ((now.reindex(old.index)-old["real"])/old["real"]*100)[m].abs().mean()
    rows.append((title, e_old, e_new))
fig_all.suptitle("Pronóstico por modelo: antes vs ahora vs real", fontsize=14)
fig_all.tight_layout(); fig_all.savefig(OUT/"resumen_4_modelos.png", dpi=110)
print(pd.DataFrame(rows, columns=["modelo", "MAPE antes ene-ago26", "MAPE ahora"]).round(2).to_string(index=False))
