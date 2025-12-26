# Usa una imagen base oficial de Python 3.12
FROM python:3.12.12-alpine3.23

WORKDIR /app

# Desactivar buffering de Python (importante para docker logs)
# Evitar creación de .pyc dentro del contenedor
ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1

# Soporte de zonas horarias (útil para tzlocal/pytz y logs)
RUN apk add --no-cache tzdata

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# Copia solo el script principal y otros archivos necesarios
COPY bot_trading_v2_2.py .
# Si tienes otros módulos o archivos necesarios, agrégalos aquí
# COPY otros_archivos_o_carpetas /app/

RUN mkdir -p /app/files /app/logs

# Ejecutar como usuario no-root (mejor práctica de seguridad)
ARG APP_UID=1000
ARG APP_GID=1000
RUN addgroup -S -g ${APP_GID} app \
 && adduser -S -D -H -u ${APP_UID} -G app app \
 && chown -R app:app /app

USER app

CMD ["python", "bot_trading_v2_2.py"]

# Notas para el usuario:
# Para desplegar imagen 
# docker run -d --name bot_trading \
#            -v /home/melgary/docker/bot-trading/files:/app/files \
#            --env-file /home/melgary/docker/bot-trading/.env \
#            melgary/bot-trading:v1.0.4
# docker run -d --name bot_trading \
#            --restart unless-stopped   
#--env-file /home/melgary/docker/bot-trading/.env   -v /home/melgary/docker/bot-trading/files:/app/files   -v /home/melgary/docker/bot-trading/logs:/app/logs   melgary/bot-trading:v2.0.5
# Para construir y subir la imagen a Docker Hub:
# docker buildx build --platform linux/amd64,linux/arm64 -t melgary/bot-trading:v2.0.0 --push .
