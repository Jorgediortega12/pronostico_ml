"""Caché en proceso para los re-entrenos/re-estimaciones en línea de los modelos v2 (ids 1, 2, 3).

En hindcast (``start`` anterior al último mes que vio el artefacto) RF/XGB se re-entrenan y SARIMAX re-estima
sus parámetros con la historia disponible. Eso cuesta 1-2 s por petición, y el front repite peticiones
idénticas (cada "Play"). Este módulo ofrece:

* :class:`SingleFlightLRU`: LRU real (``OrderedDict.move_to_end``) con tope de entradas y *single-flight* por
  clave: si dos peticiones piden la misma clave a la vez, solo una calcula y la otra espera el resultado.
* :data:`RETRAIN_SEMAPHORE`: semáforo global que limita los re-entrenos simultáneos (protege CPU/memoria del
  worker cuando llegan muchas claves distintas a la vez).
* :func:`series_digest`: hash corto (16 hex) de una Serie (índice + valores) para usar como clave.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Callable, Hashable, TypeVar

import pandas as pd

T = TypeVar("T")

#: Re-entrenos/re-estimaciones simultáneos permitidos en el proceso (todas las claves y modelos).
MAX_CONCURRENT_RETRAINS = 2
RETRAIN_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_RETRAINS)


def series_digest(s: pd.Series) -> str:
    """Hash corto y estable de una Serie (índice y valores)."""
    h = hashlib.sha1(pd.util.hash_pandas_object(s, index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


class SingleFlightLRU:
    """LRU acotado y seguro entre hilos, con cálculo único por clave (*single-flight*)."""

    def __init__(self, maxsize: int):
        if maxsize < 1:
            raise ValueError("maxsize debe ser >= 1")
        self.maxsize = int(maxsize)
        self._data: "OrderedDict[Hashable, object]" = OrderedDict()
        self._lock = threading.Lock()
        self._inflight: dict = {}

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _get_locked(self, key):
        if key in self._data:
            self._data.move_to_end(key)
            return True, self._data[key]
        return False, None

    def get_or_compute(self, key: Hashable, compute: Callable[[], T]) -> T:
        """Devuelve el valor cacheado de ``key`` o lo calcula una sola vez (las demás llamadas esperan)."""
        with self._lock:
            hit, value = self._get_locked(key)
            if hit:
                return value
            key_lock = self._inflight.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                hit, value = self._get_locked(key)
                if hit:
                    return value
            try:
                value = compute()
                with self._lock:
                    self._data[key] = value
                    self._data.move_to_end(key)
                    while len(self._data) > self.maxsize:
                        self._data.popitem(last=False)
            finally:
                with self._lock:
                    if self._inflight.get(key) is key_lock:
                        del self._inflight[key]
        return value
