# Usa una imagen base oficial de Python 3.12
FROM python:3.12-slim

WORKDIR /app

# Desactivar buffering de Python (importante para docker logs)
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia solo el script principal y otros archivos necesarios
COPY bot_trading_v2.py .
# Si tienes otros módulos o archivos necesarios, agrégalos aquí
# COPY otros_archivos_o_carpetas /app/

RUN mkdir -p /app/files /app/logs

CMD ["python", "bot_trading_v2.py"]

# Notas para el usuario:
# Para desplegar imagen 
# docker run -d --name bot_trading \
#            -v /home/melgary/docker/bot-trading/files:/app/files \ 
#            --env-file /home/melgary/docker/bot-trading/.env \
#            melgary/bot-trading:v1.0.4
# Para construir y subir la imagen a Docker Hub:
# docker buildx build --platform linux/amd64,linux/arm64 -t melgary/bot-trading:v2.0.0 --push .
