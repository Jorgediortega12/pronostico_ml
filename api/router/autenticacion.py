from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import jwt
import os

# Login simplificado de CUENTA DE SERVICIO (sin LDAP ni tabla de usuarios).
# El backend Node se autentica con ML_SERVICE_USER / ML_SERVICE_PASSWORD
# (deben coincidir con DEMANDA_ML_USER / DEMANDA_ML_PASSWORD del backend).
router = APIRouter(tags=["autenticacion"])

ALGORITHM = "HS256"


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Se leen en tiempo de request para no depender del orden de carga del .env.
    secret_key = os.getenv("JWT_SECRET_KEY")
    service_user = os.getenv("ML_SERVICE_USER")
    service_password = os.getenv("ML_SERVICE_PASSWORD")
    expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

    if not secret_key:
        raise HTTPException(status_code=500, detail="JWT_SECRET_KEY no configurado")
    if not service_user or not service_password:
        raise HTTPException(status_code=500, detail="Credenciales de servicio no configuradas")
    if form_data.username != service_user or form_data.password != service_password:
        raise HTTPException(status_code=400, detail="Credenciales inválidas")

    expire = datetime.utcnow() + timedelta(minutes=expire_minutes)
    payload = {"id": "service", "user_name": service_user, "exp": expire}
    token = jwt.encode(payload, secret_key, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}