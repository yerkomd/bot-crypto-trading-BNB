# Backtesting institucional

Este módulo ejecuta backtesting vela-por-vela reutilizando adapters de estrategias del bot productivo:

- `v2_2` → `bot_trading_v2_2.py`
- `v3_1` → `bot_trading_v3_1.py`
- `v4` → `bot_trading_v4.py`
- `v5` → `estrategia_v5.py`

Runner oficial:

```bash
python -m backtesting.run_backtest --help
```

## 1) Qué hace el engine

- Simulación **no-lookahead**:
  - señal en vela `i-1` (cerrada)
  - entrada en `open` de vela `i`
  - salidas por `stop_loss` usando `high/low` de la vela actual
- Capital compartido multi-símbolo (`cash` único).
- Fees por lado (`fee_rate`) y slippage (`slippage_bps`).
- En cierre de backtest, fuerza salida de posiciones remanentes (`FORCED_EXIT_EOD`).
- Genera:
  - `equity_curve.csv`
  - `trades.csv`
  - métricas impresas en JSON por consola.

## 2) Requisitos

- Python 3.12+.
- Dependencias de `requirements.txt`.

Preparación recomendada:

```bash
cd /home/melgary/Proyectos/data-engineer/BolsaDeValores/bot-crypto-trading-BNB
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Modos de datos (`--source`)

### 3.1) PostgreSQL (`--source postgres`)

Lee velas desde `trading.market_klines`.

Variables mínimas en `.env`:

```dotenv
USE_DATABASE=true
POSTGRES_HOST=...
POSTGRES_PORT=5432
POSTGRES_DB=...
POSTGRES_USER=...
POSTGRES_PASSWORD=...
```

Puedes cargar datos por:

- rango de fechas: `--start` + `--end`
- últimas N velas: `--lookback-bars`

### 3.2) CSV offline (`--source csv`)

- CSV mínimo: `timestamp,open,high,low,close` (opcional `volume`).
- En modo CSV solo se soporta **1 símbolo**.

## 4) Estrategias soportadas (`--strategy`)

- `v3_1` (default): EMA200 + EMA50 + RSI range + ATR-based risk levels por régimen.
- `v2_2`: RSI/StochRSI + lógica de régimen y trailing porcentual compatible.
- `v4`: breakout sobre HH(10) + EMA200, SL/trailing por ATR.
- `v5`: tendencia + ADX + filtro probabilístico ML (joblib), ATR dinámico.

Ejemplo rápido por estrategia:

```bash
python -m backtesting.run_backtest --strategy v3_1 --source postgres --symbols BTCUSDT --interval 1h --lookback-bars 2000 --output-dir backtest_out_v31
python -m backtesting.run_backtest --strategy v2_2 --source postgres --symbols BTCUSDT --interval 1h --lookback-bars 2000 --output-dir backtest_out_v22
python -m backtesting.run_backtest --strategy v4   --source postgres --symbols BTCUSDT --interval 1h --lookback-bars 2000 --output-dir backtest_out_v4
python -m backtesting.run_backtest --strategy v5   --source postgres --symbols BTCUSDT --interval 1h --lookback-bars 2000 --output-dir backtest_out_v5
```

## 5) Ejecución recomendada con config

Se soporta config en JSON o YAML:

- `configs/backtest_example.json`
- `configs/backtest_example.yaml`

Ejecutar JSON:

```bash
python -m backtesting.run_backtest --config configs/backtest_example.json
```

Ejecutar YAML:

```bash
python -m backtesting.run_backtest --config-yaml configs/backtest_example.yaml
```

Importante:

- Usa **solo uno**: `--config` o `--config-yaml`.
- Los flags de CLI **sobrescriben** el archivo de config.

## 6) Ejemplos prácticos de ejecución

### 6.1) Postgres por rango

```bash
python -m backtesting.run_backtest \
  --strategy v3_1 \
  --source postgres \
  --symbols BTCUSDT,BNBUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --initial-cash 10000 \
  --fee-rate 0.001 \
  --slippage-bps 5 \
  --max-positions-per-symbol 3 \
  --min-notional 10 \
  --output-dir backtest_out_2024_h1
```

### 6.2) Postgres por lookback

```bash
python -m backtesting.run_backtest \
  --strategy v4 \
  --source postgres \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --interval 4h \
  --lookback-bars 5000 \
  --slippage-bps 10 \
  --output-dir backtest_out_v4_4h
```

### 6.3) CSV offline

```bash
python -m backtesting.run_backtest \
  --strategy v3_1 \
  --source csv \
  --csv data/BTCUSDT_1h.csv \
  --symbols BTCUSDT \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --output-dir backtest_out_csv
```

### 6.4) Override de parámetros de estrategia desde CLI

```bash
python -m backtesting.run_backtest \
  --strategy v3_1 \
  --source postgres \
  --symbols BTCUSDT \
  --interval 1h \
  --lookback-bars 3000 \
  --rsi-min 35 --rsi-max 55 \
  --regime-params-json '{"BULL":{"BUY_COOLDOWN":7200,"POSITION_SIZE":0.08},"LATERAL":{"BUY_COOLDOWN":14400,"POSITION_SIZE":0.03},"BEAR":{"BUY_COOLDOWN":21600,"POSITION_SIZE":0.0}}' \
  --atr-multipliers-json '{"BULL":{"tp":2.5,"sl":1.5,"trailing_sl":1.2},"LATERAL":{"tp":2.0,"sl":1.2,"trailing_sl":1.0},"BEAR":null}' \
  --output-dir backtest_out_overrides
```

### 6.5) Configs ya incluidos para ventanas amplias

```bash
python -m backtesting.run_backtest --config-yaml configs/backtest_2021.yaml
python -m backtesting.run_backtest --config-yaml configs/backtest_2021-2025.yaml
python -m backtesting.run_backtest --config-yaml configs/backtest_2021-2025_4h.yaml
```

## 7) Parámetros principales del runner

### Motor

- `--initial-cash`: capital inicial.
- `--fee-rate`: fee por lado (default `0.001` = 0.1%).
- `--slippage-bps`: slippage en bps (5 = 0.05%).
- `--max-positions-per-symbol`: máximo de posiciones por símbolo.
- `--min-notional`: mínimo notional por entrada.
- `--hard-cooldown-s`: cooldown adicional forzado por engine.

### Datos

- `--source`: `postgres` o `csv`.
- `--symbols`: lista separada por comas.
- `--interval`: `1m`, `15m`, `1h`, `4h`, `1d`, etc.
- `--start`, `--end`: rango (ISO o epoch s/ms).
- `--lookback-bars`: últimas N velas (postgres).
- `--limit`: cap opcional en consultas por rango.
- `--csv`: ruta CSV cuando `--source csv`.

### Estrategia

- `--strategy`: `v2_2`, `v3_1`, `v4`.
- `--strategy`: `v2_2`, `v3_1`, `v4`, `v5`.
- `--rsi-min`, `--rsi-max` (aplica a `v3_1`).
- `--regime-params-json` (aplica a adapters que usan régimen).
- `--atr-multipliers-json` (aplica a `v3_1`).
- `--v5-threshold-override` (aplica a `v5`, para backtesting/control de sensibilidad).
- `--v5-model-path` (aplica a `v5`, ruta al artefacto joblib).

### Salida

- `--output-dir`: carpeta de resultados.

## 8) Archivos de salida

En `--output-dir`:

- `equity_curve.csv`
  - `timestamp,cash,positions_value,equity,open_positions,drawdown`
- `trades.csv`
  - `symbol,entry_time,exit_time,qty,entry_price,exit_price,reason,pnl,pnl_pct,fees_paid,duration_bars`

Además, el runner imprime en consola métricas en JSON:

- `sharpe`
- `sortino`
- `max_drawdown`
- `profit_factor`
- `expectancy`
- `calmar`
- `win_rate`
- `avg_trade_duration_s`

## 9) Troubleshooting rápido

- Error: `For --source=postgres you must provide --start and --end, or --lookback-bars`
  - Solución: pasa un rango o `--lookback-bars`.
- Error: `--csv is required when --source=csv`
  - Solución: agrega `--csv path/al/archivo.csv`.
- Error: `--source=csv supports exactly one symbol`
  - Solución: usa un solo símbolo en `--symbols`.
- Error: `No market data loaded from Postgres`
  - Revisa `POSTGRES_*`, datos en `trading.market_klines`, símbolo/intervalo y ventana temporal.
- Error con YAML config
  - Verifica que `PyYAML` esté instalado (`requirements.txt` ya lo incluye).

## 10) Script legacy

El repo conserva `backtest_strategy.py` (flujo anterior). Para el flujo institucional y consistente con adapters de producción, usa:

```bash
python -m backtesting.run_backtest
```
