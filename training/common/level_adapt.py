"""Correccion adaptativa de NIVEL para pronosticos mensuales de demanda (multiplicativa, amortiguada).

Problema: ante un cambio de nivel (p. ej. el crecimiento interanual pasa de ~4 % a ~11 %) los modelos
RF/XGB/Clima siguen subestimando varios meses aunque ya haya meses observados del nuevo regimen. Este modulo
estima el sesgo de nivel RECIENTE del propio modelo con residuos FUERA DE MUESTRA y lo aplica, amortiguado,
a los meses pronosticados.

Procedimiento (funcion pura, sin I/O; ver :func:`adjust`):

1. ``last`` = ultimo mes observado de la historia (todo lo anterior a ``start``). El **origen de referencia** es
   ``last - k`` meses: se pronostica con el MISMO proceso de inferencia reajustado solo con datos ``<=`` ese
   origen (``ref_fn``), de modo que los ``k`` meses siguientes (hasta ``last``) son pronostico fuera de muestra.
2. Residuos ``e_m = log(real_m / pronostico_ref_m)`` de esos meses (los meses de exclusiones, p. ej. COVID, y los
   no finitos se omiten). Resumen ``L`` = media (o mediana) ponderada exponencialmente (semivida ``hl_w`` meses;
   mas peso a los meses recientes).
3. Ajuste en log ``c = clip(phi * L, -cap, cap)`` y factor por horizonte ``h`` (meses desde ``last``)
   ``f_h = exp(c * 0.5 ** ((h - 1) / hl_h))``: amortiguado por ``phi`` y decreciente con el horizonte.
4. Sin residuos validos (menos de ``min_obs``) o sin pronostico de referencia -> factor 1 (sin cambio).

No hay fuga: el pronostico de referencia y los residuos solo usan datos ``<= last``; el factor solo escala
meses ``> last``. Los hiperparametros (``LevelSpec``) se eligen con orígenes <= 2023-12 (ver docs/work/11).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from training.common import config

Exclusion = tuple[pd.Timestamp, Optional[pd.Timestamp]]
#: ``ref_fn(historia_hasta_origen_ref, horizonte) -> Serie`` pronostico (kWh) indexada por mes.
RefFn = Callable[[pd.Series, int], pd.Series]


@dataclass(frozen=True)
class LevelSpec:
    k: int = 6  # meses de residuos (= distancia del origen de referencia a ``last``)
    hl_w: Optional[float] = 3.0  # semivida (meses) de los pesos de los residuos; None = pesos iguales
    est: str = "mean"  # "mean" | "median"
    phi: float = 0.5  # amortiguacion del nivel estimado (0 = apagado, 1 = completo)
    hl_h: Optional[float] = 12.0  # semivida (meses) del factor segun el horizonte; None = sin decaimiento
    cap: float = 0.10  # |ajuste| maximo en log (~ +-10 %)
    min_obs: int = 3  # residuos validos minimos; menos -> factor 1
    tau: float = 0.0  # zona muerta (log): L_ef = sign(L) * max(|L| - tau, 0) (ruido normal no se corrige)
    agree: float = 0.0  # fraccion minima de residuos con el signo de L; menos -> sin ajuste (desvio no persistente)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LevelSpec":
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})

    def validate(self) -> None:
        if self.k < 1 or self.min_obs < 1 or self.min_obs > self.k:
            raise ValueError("LevelSpec: k >= 1 y 1 <= min_obs <= k")
        if self.est not in ("mean", "median"):
            raise ValueError("LevelSpec.est debe ser 'mean' o 'median'")
        if not 0.0 <= self.phi <= 1.0 or self.cap < 0 or self.tau < 0 or not 0.0 <= self.agree <= 1.0:
            raise ValueError("LevelSpec: 0 <= phi <= 1, cap >= 0, tau >= 0 y 0 <= agree <= 1")
        if (self.hl_w is not None and self.hl_w <= 0) or (self.hl_h is not None and self.hl_h <= 0):
            raise ValueError("LevelSpec: las semividas deben ser > 0 (o None)")


@dataclass
class LevelResult:
    """Resultado del ajuste. ``factor`` esta indexado por los meses pronosticados (1.0 si no hay ajuste)."""

    factor: pd.Series
    level: float = 0.0  # L: nivel estimado (log), antes de phi/cap/decaimiento
    ajuste_log: float = 0.0  # c = clip(phi * L, +-cap)
    n_obs: int = 0
    origen_ref: Optional[pd.Timestamp] = None
    aplicado: bool = False
    motivo: Optional[str] = None  # por que no se aplico (None si se aplico)

    def to_dict(self) -> dict:
        f = self.factor
        return {
            "aplicado": bool(self.aplicado),
            "motivo": self.motivo,
            "factor_primer_mes": float(f.iloc[0]) if len(f) else 1.0,
            "factor_ultimo_mes": float(f.iloc[-1]) if len(f) else 1.0,
            "nivel_residual_pct": float((np.exp(self.level) - 1.0) * 100.0),
            "ajuste_pct": float((np.exp(self.ajuste_log) - 1.0) * 100.0),
            "meses_residuo": int(self.n_obs),
            "origen_referencia": f"{self.origen_ref:%Y-%m}" if self.origen_ref is not None else None,
        }


def _month(x) -> pd.Timestamp:
    return pd.Timestamp(x).to_period("M").to_timestamp()


def _excluded(idx: pd.DatetimeIndex, exclusions: Sequence[Exclusion]) -> np.ndarray:
    mask = np.zeros(len(idx), dtype=bool)
    for lo, hi in exclusions:
        m = np.asarray(idx >= lo)
        if hi is not None:
            m &= np.asarray(idx <= hi)
        mask |= m
    return mask


def covid_exclusions() -> list[Exclusion]:
    """Meses COVID de ``training.common.config`` (no se usan como residuos: no representan nivel)."""
    return [(_month(config.COVID_EXCLUDED[0]), _month(config.COVID_EXCLUDED[1]))]


def estimate_level(resid: pd.Series, last: pd.Timestamp, spec: LevelSpec) -> tuple[float, int]:
    """``(L, n)``: resumen ponderado de los residuos log (con zona muerta ``tau`` y compuerta de persistencia
    ``agree``) y cuantos validos hubo (``L = 0`` si ``n < min_obs``)."""
    e = pd.Series(resid, dtype=float).dropna()
    e = e[np.isfinite(e.to_numpy())]
    n = int(len(e))
    if n < spec.min_obs:
        return 0.0, n
    age = np.array([(last.year - t.year) * 12 + last.month - t.month for t in e.index], dtype=float)
    w = np.ones(n) if spec.hl_w is None else 0.5 ** (age / spec.hl_w)
    v = e.to_numpy(dtype=float)
    if spec.est == "mean":
        lv = float(np.sum(w * v) / np.sum(w))
    else:
        order = np.argsort(v)
        cw = np.cumsum(w[order])
        lv = float(v[order][np.searchsorted(cw, 0.5 * cw[-1])])
    if spec.agree > 0 and float(np.mean(np.sign(v) == np.sign(lv))) < spec.agree - 1e-12:
        return 0.0, n  # el desvio no es persistente (residuos de signo mixto)
    return float(np.sign(lv) * max(abs(lv) - spec.tau, 0.0)), n


def horizon_decay(h: np.ndarray, spec: LevelSpec) -> np.ndarray:
    """Peso por horizonte ``h >= 1`` (meses desde ``last``): 1 en ``h = 1`` y mitad cada ``hl_h`` meses."""
    h = np.asarray(h, dtype=float)
    if spec.hl_h is None:
        return np.ones_like(h)
    return 0.5 ** (np.maximum(h - 1.0, 0.0) / spec.hl_h)


def factors_from_level(level: float, horizons: np.ndarray, spec: LevelSpec) -> tuple[np.ndarray, float]:
    """``(factores, c)`` con ``c = clip(phi * level, +-cap)`` y ``factor_h = exp(c * decay(h))``."""
    c = float(np.clip(spec.phi * level, -spec.cap, spec.cap))
    return np.exp(c * horizon_decay(horizons, spec)), c


def residuals(hist: pd.Series, ref_pred: pd.Series, exclusions: Sequence[Exclusion] = ()) -> pd.Series:
    """``log(real / pronostico_ref)`` en los meses con ambos datos y fuera de las exclusiones."""
    common = hist.index.intersection(ref_pred.index)
    common = common[~_excluded(pd.DatetimeIndex(common), exclusions)]
    y = hist.reindex(common).to_numpy(dtype=float)
    p = ref_pred.reindex(common).to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    return pd.Series(np.log(y[ok] / p[ok]), index=common[ok])


def adjust(
    hist: pd.Series,
    forecast: pd.Series,
    ref_fn: RefFn,
    spec: LevelSpec,
    exclusions: Sequence[Exclusion] = (),
) -> LevelResult:
    """Calcula el factor de nivel para ``forecast`` (meses posteriores al ultimo de ``hist``).

    ``hist``: reales observados (indice inicio de mes) usados por el pronostico base; ``forecast``: pronostico base
    (kWh) desde ``hist.index.max()+1``; ``ref_fn(hist_ref, h)``: el mismo proceso de inferencia reajustado con
    ``hist_ref`` (solo meses ``<=`` origen de referencia), que devuelve ``h`` meses desde ese origen. Cualquier
    error de ``ref_fn`` (historia corta, etc.) deja el factor en 1 con ``motivo`` explicativo.
    """
    spec.validate()
    one = pd.Series(1.0, index=forecast.index)
    if hist.empty or forecast.empty:
        return LevelResult(one, motivo="sin_historia_o_pronostico")
    last = _month(hist.index.max())
    fidx = pd.DatetimeIndex(forecast.index)
    if (fidx <= last).any():
        raise ValueError("el pronostico debe ser posterior al ultimo mes de la historia")
    origin_ref = last - pd.DateOffset(months=spec.k)
    hist_ref = hist[hist.index <= origin_ref]
    if hist_ref.empty:
        return LevelResult(one, origen_ref=origin_ref, motivo="sin_historia_en_origen_ref")
    try:
        ref = ref_fn(hist_ref.copy(), spec.k)
    except Exception as exc:  # historia insuficiente para reajustar, etc.: nunca rompe el pronostico
        return LevelResult(one, origen_ref=origin_ref, motivo=f"ref_fn: {type(exc).__name__}")
    ref = ref[(ref.index > origin_ref) & (ref.index <= last)]
    e = residuals(hist, ref, exclusions)
    level, n = estimate_level(e, last, spec)
    if n < spec.min_obs:
        return LevelResult(one, origen_ref=origin_ref, n_obs=n, motivo="residuos_insuficientes")
    h = np.array([(t.year - last.year) * 12 + t.month - last.month for t in fidx])
    f, c = factors_from_level(level, h, spec)
    return LevelResult(pd.Series(f, index=forecast.index), level=level, ajuste_log=c, n_obs=n,
                       origen_ref=origin_ref, aplicado=True)


def apply_level(forecast: pd.Series, result: LevelResult) -> pd.Series:
    """Pronostico corregido = ``forecast * factor`` (alineado por indice)."""
    return forecast * result.factor.reindex(forecast.index).fillna(1.0)
