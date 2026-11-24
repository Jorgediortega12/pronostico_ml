"""Pronostico con start a mitad de año (conoce los meses previos): sin vs con LEVEL_ADAPT=1, vs real y baseline de ayer."""
import importlib.util, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; OUT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("bl", ROOT/"docs/work/00_entorno/scripts/baseline.py")
bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
db = bl.SessionLocal(); real = bl.real_monthly(db)
old = pd.read_csv(ROOT/"docs/work/00_entorno/baseline_2026_2027.csv", index_col="mes"); old.index = pd.PeriodIndex(old.index, freq="M")
START, END = "2026-07-01", "2026-08-31"
per = pd.period_range("2026-01", "2026-08", freq="M")
fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex=True); rows = []
for ax, (mid, name) in zip(axs.ravel(), {1: "Random Forest", 2: "XGBoost", 3: "SARIMAX", 4: "Climático + ONI"}.items()):
    res = {}
    for flag in ("0", "1"):
        os.environ["LEVEL_ADAPT"] = flag
        s, _ = bl.run_model(db, mid, START, END, raw=True); res[flag] = s
    os.environ.pop("LEVEL_ADAPT", None)
    x = per.to_timestamp()
    ax.plot(x, real.reindex(per)/1e6, "k-o", lw=2, ms=4, label="Real")
    k = bl.MODELS[mid]
    ax.plot(x, old[f"{k}_crudo"].reindex(per)/1e6, "--", color="#d62728", label="Baseline de ayer (start 2026-01)")
    ax.plot(res["0"].reindex(per).dropna().index.to_timestamp(), res["0"].reindex(per).dropna()/1e6, "-o", color="#2ca02c", ms=4, label="Ahora, sin corrección (start 2026-07)")
    ax.plot(res["1"].reindex(per).dropna().index.to_timestamp(), res["1"].reindex(per).dropna()/1e6, "-s", color="#1f77b4", ms=4, label="Con LEVEL_ADAPT=1 (start 2026-07)")
    ax.axvline(pd.Timestamp("2026-06-20"), color="#999", lw=.8); ax.set_title(name); ax.set_ylabel("GWh/mes"); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper left")
    for f in ("0", "1"):
        m = real.reindex(pd.period_range("2026-07", "2026-08", freq="M"))
        e = ((res[f].reindex(m.index)-m)/m*100).abs().mean(); rows.append((name, "sin" if f == "0" else "con", round(e, 2)))
fig.suptitle("Pronóstico con start 2026-07 (conoce ene-jun): sin vs con corrección de nivel", fontsize=14)
fig.tight_layout(); fig.savefig(OUT/"nivel_adaptativo_start_2026_07.png", dpi=110)
print(pd.DataFrame(rows, columns=["modelo", "corr", "MAPE jul-ago"]).to_string(index=False))
