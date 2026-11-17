"""La base de datos del ML se elige por petición (database_url) y no solo desde el .env del servidor."""
import asyncio
import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import api.config.database as dbmod

URL_A = "postgresql://usr_a:clave_secreta_A@servidor1:5432/empresa_a"
URL_B = "postgresql://usr_b:clave_secreta_B@servidor1:5432/empresa_b"


def _req(metodo="POST", cuerpo=None, headers=None, query=b""):
    hs = {k.lower(): v for k, v in (headers or {}).items()}
    if cuerpo is not None:
        hs.setdefault("content-type", "application/json")
        data = json.dumps(cuerpo).encode()
    else:
        data = b""
    scope = {
        "type": "http", "method": metodo, "path": "/x", "query_string": query,
        "headers": [(k.encode(), v.encode()) for k, v in hs.items()],
    }

    async def receive():
        return {"type": "http.request", "body": data, "more_body": False}

    return Request(scope, receive)


def _correr(coro):
    return asyncio.run(coro)


class _MotorFalso:
    def __init__(self, url, **kw):
        self.url = url
        self.kw = kw

    def dispose(self):
        pass


@pytest.fixture(autouse=True)
def motores_falsos(monkeypatch):
    creados = []

    def fabrica(url, **kw):
        m = _MotorFalso(url, **kw)
        creados.append(m)
        return m

    monkeypatch.setattr(dbmod, "create_engine", fabrica)
    monkeypatch.setattr(dbmod, "_motores", dbmod.OrderedDict())
    monkeypatch.delenv("ML_ALLOWED_DB_HOSTS", raising=False)
    monkeypatch.delenv("ML_REQUIRE_DATABASE_URL", raising=False)
    return creados


def _obtener_bind(request):
    async def usar():
        gen = dbmod.get_db(request)
        db = await gen.__anext__()
        bind = db.get_bind()
        await gen.aclose()
        return bind

    return _correr(usar())


def test_url_del_cuerpo_de_un_post():
    assert _correr(dbmod._url_de_la_peticion(_req(cuerpo={"database_url": URL_A}))) == URL_A


def test_url_del_parametro_de_un_get():
    assert _correr(dbmod._url_de_la_peticion(_req("GET", query=b"database_url=" + URL_B.encode()))) == URL_B


def test_prioridad_encabezado_sobre_cuerpo():
    r = _req(cuerpo={"database_url": URL_A}, headers={"X-Database-Url": URL_B})
    assert _correr(dbmod._url_de_la_peticion(r)) == URL_B


def test_sin_url_devuelve_none():
    assert _correr(dbmod._url_de_la_peticion(_req(cuerpo={"otro": 1}))) is None
    assert _correr(dbmod._url_de_la_peticion(_req("GET"))) is None


def test_normaliza_el_esquema():
    assert dbmod._normalizar_url("postgresql://u:p@h:5432/d").startswith("postgresql+psycopg2://")
    assert dbmod._normalizar_url("postgres://u:p@h:5432/d").startswith("postgresql+psycopg2://")


@pytest.mark.parametrize("url", ["mysql://u:p@h/d", "postgresql://u:p@h:5432/", "postgresql:///d", "texto"])
def test_rechaza_urls_invalidas(url):
    with pytest.raises(ValueError):
        dbmod._validar_url(dbmod._normalizar_url(url))


def test_una_base_distinta_por_empresa_y_se_reutiliza(motores_falsos):
    bind_a1 = _obtener_bind(_req(cuerpo={"database_url": URL_A}))
    bind_b = _obtener_bind(_req(cuerpo={"database_url": URL_B}))
    bind_a2 = _obtener_bind(_req(cuerpo={"database_url": URL_A}))
    assert bind_a1.url.endswith("/empresa_a")
    assert bind_b.url.endswith("/empresa_b")
    assert bind_a1 is bind_a2              # mismo motor para la misma base
    assert len(motores_falsos) == 2        # solo se crearon dos motores


def test_sin_database_url_usa_la_base_por_defecto():
    bind = _obtener_bind(_req(cuerpo={"otro": 1}))
    assert bind is dbmod.engine


def test_exigir_database_url(monkeypatch):
    monkeypatch.setenv("ML_REQUIRE_DATABASE_URL", "1")
    with pytest.raises(HTTPException) as e:
        _obtener_bind(_req(cuerpo={"otro": 1}))
    assert e.value.status_code == 400


def test_lista_de_servidores_permitidos(monkeypatch):
    monkeypatch.setenv("ML_ALLOWED_DB_HOSTS", "servidor1:5432, localhost:5432")
    assert _obtener_bind(_req(cuerpo={"database_url": URL_A})).url.endswith("/empresa_a")
    with pytest.raises(HTTPException) as e:
        _obtener_bind(_req(cuerpo={"database_url": "postgresql://u:p@otro:5432/d"}))
    assert e.value.status_code == 400


def test_el_error_no_filtra_la_contrasena(monkeypatch):
    monkeypatch.setenv("ML_ALLOWED_DB_HOSTS", "solo-este:5432")
    with pytest.raises(HTTPException) as e:
        _obtener_bind(_req(cuerpo={"database_url": URL_A}))
    assert "clave_secreta_A" not in str(e.value.detail)
    assert "usr_a" not in str(e.value.detail)


def test_se_descartan_los_motores_menos_usados(monkeypatch, motores_falsos):
    monkeypatch.setattr(dbmod, "_MAX_MOTORES", 2)
    for i in range(4):
        _obtener_bind(_req(cuerpo={"database_url": f"postgresql://u:p@h:5432/base{i}"}))
    assert len(dbmod._motores) == 2
