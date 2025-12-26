# Pruebas de regresión

Estas pruebas están pensadas para ejecutarse con el mismo intérprete/venv que usa el bot (para tener instalados `pandas`, `python-binance`, `ta`, etc.).

Desde la carpeta del bot:

- Ejecutar todo:
  - `/home/melgary/Proyectos/data-engineer/BolsaDeValores/python-code/bot-trading/bin/python -m unittest discover -s tests -v`

Notas:
- Si usas `python3` del sistema y no tienes dependencias instaladas, verás errores de importación (por ejemplo `pandas` o `binance`).
