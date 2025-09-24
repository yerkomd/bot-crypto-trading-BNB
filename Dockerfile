# Usa una imagen base oficial de Python 3.12
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia solo el script principal y otros archivos necesarios
COPY bot-traiding.py .
# Si tienes otros módulos o archivos necesarios, agrégalos aquí
# COPY otros_archivos_o_carpetas /app/

RUN mkdir -p /app/files

CMD ["python", "bot-traiding.py"]

# Notas para el usuario:
# - Monta el archivo de variables de entorno, logs y posiciones como volúmenes externos al correr el contenedor:
# docker run -v /ruta/local/.env:/app/.env \
#            -v /ruta/local/logs:/app/logs \
#            -v /ruta/local/posiciones:/app/posiciones \
#            --env-file /app/.env \
#            nombre_imagen
