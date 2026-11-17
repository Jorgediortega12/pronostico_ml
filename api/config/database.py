import json
import logging
import os
import threading
from collections import OrderedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import HTTPException, Request
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

# ── Base POR DEFECTO del servicio ──────────────────────────────────────────────────────────────────────────────────────
# Se arma desde ML_DATABASE_URL o desde las variables POSTGRES_* (las mismas que usa analysis.py vía psycopg2). Solo se
# usa cuando la petición NO indica su propia base (desarrollo local, pruebas, scripts); ver get_db().
DATABASE_URL = os.getenv("ML_DATABASE_URL")
if not DATABASE_URL:
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB", "postgres")
    DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_timeout=60,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ── Base POR PETICIÓN (multi-empresa) ──────────────────────────────────────────────────────────────────────────────────
# Una sola instancia del ML atiende a todas las empresas: cada petición del backend trae el `database_url` de la empresa
# con la que inició sesión el usuario (en el cuerpo de los POST, en el parámetro `database_url` de los GET o en el
# encabezado X-Database-Url). Antes se ignoraba y todo caía en la base del .env del servidor.
#
#   ML_ALLOWED_DB_HOSTS     (opcional) lista "host:puerto,host:puerto" de servidores permitidos; si se define, cualquier
#                           otro se rechaza. Recomendado en producción.
#   ML_REQUIRE_DATABASE_URL (opcional) "1" para rechazar las peticiones sin database_url en vez de usar la base del .env.
_MAX_MOTORES = int(os.getenv("ML_MAX_DB_ENGINES", "32"))
_motores: "OrderedDict[str, sessionmaker]" = OrderedDict()
_motores_lock = threading.Lock()
_aviso_base_por_defecto = False


def _normalizar_url(url: str) -> str:
    u = url.strip()
    if u.startswith("postgres://"):
        u = "postgresql://" + u[len("postgres://"):]
    if u.startswith("postgresql://"):
        u = "postgresql+psycopg2://" + u[len("postgresql://"):]
    return u


def _hosts_permitidos():
    crudo = os.getenv("ML_ALLOWED_DB_HOSTS", "").strip()
    if not crudo:
        return None
    return {h.strip().lower() for h in crudo.split(",") if h.strip()}


def _validar_url(url: str) -> None:
    """Lanza ValueError (sin incluir la URL, que trae usuario y contraseña) si no es una URL de Postgres permitida."""
    p = urlparse(url)
    if p.scheme not in ("postgresql", "postgresql+psycopg2"):
        raise ValueError("solo se aceptan bases PostgreSQL")
    if not p.hostname or not p.path.strip("/"):
        raise ValueError("falta el servidor o el nombre de la base")
    permitidos = _hosts_permitidos()
    if permitidos is not None and f"{p.hostname}:{p.port or 5432}".lower() not in permitidos:
        raise ValueError("servidor de base de datos no permitido")


def _sessionmaker_para(url: str) -> sessionmaker:
    """Un motor (con su pool) por base de datos, reutilizado entre peticiones."""
    with _motores_lock:
        maker = _motores.get(url)
        if maker is not None:
            _motores.move_to_end(url)
            return maker
        eng = create_engine(url, pool_size=5, max_overflow=10, pool_timeout=60, pool_pre_ping=True, pool_recycle=1800)
        maker = sessionmaker(autocommit=False, autoflush=False, bind=eng)
        _motores[url] = maker
        while len(_motores) > _MAX_MOTORES:  # se descarta el menos usado
            _, viejo = _motores.popitem(last=False)
            try:
                viejo.kw["bind"].dispose()
            except Exception:  # pragma: no cover - limpieza best-effort
                pass
        p = urlparse(url)
        logger.info("ML: motor de base de datos creado para %s:%s/%s", p.hostname, p.port or 5432, p.path.strip("/"))
        return maker


async def _url_de_la_peticion(request: Request):
    url = request.headers.get("x-database-url") or request.query_params.get("database_url")
    if url:
        return url
    if request.method in ("POST", "PUT", "PATCH") and "json" in (request.headers.get("content-type") or "").lower():
        try:
            cuerpo = json.loads((await request.body()) or b"{}")
        except ValueError:
            return None
        if isinstance(cuerpo, dict) and isinstance(cuerpo.get("database_url"), str):
            return cuerpo["database_url"]
    return None


async def get_db(request: Request):
    global _aviso_base_por_defecto
    url = await _url_de_la_peticion(request)
    if url:
        url = _normalizar_url(url)
        try:
            _validar_url(url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"database_url no válida: {e}")
        maker = _sessionmaker_para(url)
    else:
        if os.getenv("ML_REQUIRE_DATABASE_URL", "").strip().lower() in ("1", "true", "si", "sí"):
            raise HTTPException(status_code=400, detail="La petición no indica la base de datos (database_url).")
        if not _aviso_base_por_defecto:
            logger.warning("ML: petición sin database_url; se usa la base por defecto del .env (ML_DATABASE_URL / POSTGRES_*).")
            _aviso_base_por_defecto = True
        maker = SessionLocal
    db = maker()
    try:
        yield db
    finally:
        db.close()
