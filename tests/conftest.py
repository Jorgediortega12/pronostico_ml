"""Configuración compartida de pytest para la suite del backend GMR.

Asegura que el paquete ``api`` sea importable al ejecutar pytest desde
cualquier directorio.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
