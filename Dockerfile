# Utilizar la imagen base de Python
FROM python:3.12.8-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y libaio1


# Establecer el directorio de trabajo
WORKDIR /api

# Copiar los archivos necesarios
COPY ./requirements.txt /api/requirements.txt
RUN pip install --no-cache-dir -r /api/requirements.txt

# Copiar el código fuente
COPY ./api /api/api

# Cargar las variables de entorno desde el archivo .env
COPY .env /api/

COPY ./instantclient_11_2 /api/instantclient_11_2


ENV LD_LIBRARY_PATH=/api/instantclient_11_2
ENV ENV=production

# Exponer el puerto donde correrá FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación y tareas en segundo plano
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]