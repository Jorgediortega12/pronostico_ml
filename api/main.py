from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Cargar variables de entorno ANTES de importar routers/servicios
# (varios leen os.getenv a nivel de módulo).
load_dotenv()

# Microservicio ML: solo expone el cómputo que el backend Node (pronostico) consume.
from .router import autenticacion, forecast, mpm

# NOTA: este servicio NO crea tablas (create_all). El esquema lo administra el
# proyecto "pronostico" (Node). Aquí solo se leen/escriben las tablas ya existentes.

IS_PRODUCTION = os.getenv("ENV") == "production"

app = FastAPI(
    title="Pronostico ML Service",
    description="Servicio de cómputo ML (forecast + MPM) para el backend de pronóstico.",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Errores de validación sin el cuerpo enviado.

    El `input` que FastAPI incluye por defecto trae `database_url` (usuario y contraseña de la base de
    datos del cliente) y no debe viajar de vuelta al navegador.
    """
    detail = [{"type": e.get("type"), "loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


app.include_router(autenticacion.router, prefix="/v1/autenticacion")
app.include_router(forecast.router, prefix="/v1/forecast")
app.include_router(mpm.router, prefix="/v1/mpm")


@app.get("/")
async def root():
    return {"message": "Pronostico ML Service OK"}