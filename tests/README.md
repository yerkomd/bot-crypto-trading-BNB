# Tests del proyecto

Este directorio contiene pruebas unitarias y de regresión para estrategia, arquitectura y capas de riesgo.

Se recomienda ejecutar siempre con el mismo entorno virtual del bot:

- `/home/melgary/Proyectos/.venv/bin/python`

## Comandos base

Desde la carpeta raíz del bot:

- Ejecutar todos los tests (`unittest`):
  - `/home/melgary/Proyectos/.venv/bin/python -m unittest discover -s tests -v`

- Ejecutar todos los tests (`pytest`, opcional):
  - `/home/melgary/Proyectos/.venv/bin/python -m pytest tests -q`

## Suites disponibles

- `tests/test_backtesting_bot_v2_2_strategy.py`
  - Backtesting y regresión de estrategia v2.2.
- `tests/test_backtesting_bot_v3_1_strategy.py`
  - Backtesting y regresión de estrategia v3.1.
- `tests/test_backtesting_bot_v4_strategy.py`
  - Backtesting y regresión de estrategia v4.
- `tests/test_quantity_utils.py`
  - Utilidades de cantidad (`floor_to_step`, saneamiento y validaciones relacionadas).
- `tests/test_regression_regime_and_logging.py`
  - Regresión de régimen de mercado y logging operativo.
- `tests/test_v3_1_architecture_and_regression.py`
  - Arquitectura v3.1 y regresiones críticas de integración.
- `tests/test_risk_layer_v2.py`
  - Unit tests e integración mínima de `risk_layer_v2`.
- `tests/test_risk_layer_v3.py`
  - Unit tests de `risk_layer_v3` (sizer, correlación, VaR, slippage y equity regime).

## Comandos por suite

### Backtesting

- `/home/melgary/Proyectos/.venv/bin/python -m unittest tests.test_backtesting_bot_v2_2_strategy -v`
- `/home/melgary/Proyectos/.venv/bin/python -m unittest tests.test_backtesting_bot_v3_1_strategy -v`
- `/home/melgary/Proyectos/.venv/bin/python -m unittest tests.test_backtesting_bot_v4_strategy -v`

### Arquitectura y regresión

- `/home/melgary/Proyectos/.venv/bin/python -m unittest tests.test_v3_1_architecture_and_regression -v`
- `/home/melgary/Proyectos/.venv/bin/python -m unittest tests.test_regression_regime_and_logging -v`

### Utilidades

- `/home/melgary/Proyectos/.venv/bin/python -m unittest tests.test_quantity_utils -v`

### Risk Layers (v2 y v3)

- `/home/melgary/Proyectos/.venv/bin/python -m unittest tests.test_risk_layer_v2 tests.test_risk_layer_v3 -v`

Equivalente con `pytest` (si está instalado):

- `/home/melgary/Proyectos/.venv/bin/python -m pytest tests/test_risk_layer_v2.py tests/test_risk_layer_v3.py -q`

## Atajos útiles

- Ejecutar solo tests de riesgo (`pytest` por patrón):
  - `/home/melgary/Proyectos/.venv/bin/python -m pytest tests/test_risk_layer_*.py -q`

- Ejecutar solo backtesting (`pytest` por patrón):
  - `/home/melgary/Proyectos/.venv/bin/python -m pytest tests/test_backtesting_*.py -q`

## Notas

- Si usas `python3` del sistema sin dependencias, verás errores de importación (`pandas`, `binance`, `ta`, etc.).
- Si no tienes `pytest` instalado, usa `unittest` (estándar de Python).
