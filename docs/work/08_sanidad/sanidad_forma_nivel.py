"""Sanidad de FORMA (perfil estacional) y NIVEL (totales anuales / crecimiento) de los 4 modelos.
Corre la API real (modo crudo, sin tope/override), horizonte 2026-01..2029-12, y compara contra el historico real
(escala 100 % mercado; meses post-division 2026-09+ se excluyen del real porque estan en 70 %).
Uso: python docs/work/08_sanidad/sanidad_forma_nivel.py"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("bl", ROOT / "docs/work/00_entorno/scripts/baseline.py")
bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
db = bl.SessionLocal(); real = bl.real_monthly(db)
real_full = real[real.index <= pd.Period("2026-08", "M")]
# perfil estacional historico: promedio de share mensual en anios completos 2023-2025
def shares(s):
    out = {}
    for y in sorted(set(s.index.year)):
        m = s[s.index.year == y]
        if len(m) == 12: out[y] = (m / m.sum()).values
    return out
hs = shares(real_full); hist_years = [y for y in (2023, 2024, 2025) if y in hs]
hist_profile = np.mean([hs[y] for y in hist_years], axis=0)
ann = real_full.groupby(real_full.index.year).sum()
print("Anual real (GWh):", {int(y): round(v/1e6,1) for y, v in ann.items() if y >= 2019})
print("Crecimiento real YoY %:", {int(y): round((ann[y]/ann[y-1]-1)*100,2) for y in ann.index if y-1 in ann.index and y >= 2022 and y <= 2025})
print("Perfil estacional historico 2023-25 (%):", np.round(hist_profile*100,2).tolist())
for mid, name in bl.MODELS.items():
    s, resp = bl.run_model(db, mid, "2026-01-01", "2029-12-31", raw=True)
    print(f"\n=== {name} (id {mid}) ===")
    a = s.groupby(s.index.year).sum()
    print("Anual pron (GWh):", {int(y): round(v/1e6,1) for y, v in a.items()})
    g = {int(y): round((a[y]/a[y-1]-1)*100,2) for y in a.index if y-1 in a.index}
    g[2026] = round((a[2026]/ann[2025]-1)*100,2)
    print("Crecimiento YoY % (2026 vs real 2025):", dict(sorted(g.items())))
    for y in (2027, 2028):
        m = s[s.index.year == y]
        if len(m) == 12:
            sh = (m/m.sum()).values
            corr = np.corrcoef(sh, hist_profile)[0, 1]
            print(f"  {y}: corr forma vs hist={corr:.3f}  pico mes {int(np.argmax(sh))+1} (hist {int(np.argmax(hist_profile))+1}) "
                  f"valle mes {int(np.argmin(sh))+1} (hist {int(np.argmin(hist_profile))+1}) amplitud pron={sh.max()/sh.min():.3f} hist={hist_profile.max()/hist_profile.min():.3f} "
                  f"max|dif share| pp={np.abs(sh-hist_profile).max()*100:.2f}")
    m26 = s[s.index.year == 2026]
    r26 = real_full[real_full.index.year == 2026]
    print("  Ene-Ago 2026 pron vs real (GWh):", [(round(m26[p]/1e6,1), round(r26[p]/1e6,1)) for p in r26.index])
    mom = s.pct_change().dropna()*100
    print(f"  salto mensual max {mom.max():.1f}% / min {mom.min():.1f}%  (hist max {real_full.pct_change().dropna().max()*100:.1f}% min {real_full.pct_change().dropna().min()*100:.1f}% excl. sin filtro)")
    print("  rango mensual pron GWh:", round(s.min()/1e6,1), "-", round(s.max()/1e6,1), "| hist 2023-25:", round(real_full[real_full.index.year>=2023].min()/1e6,1), "-", round(real_full[real_full.index.year>=2023].max()/1e6,1))
