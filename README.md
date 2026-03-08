
# Bot Crypto Trading v5 — Binance Spot (Multi-Strategy)

Sistema de trading algorítmico para Binance Spot con arquitectura multi-estrategia,
capas de riesgo institucional y persistencia total en PostgreSQL.

---

## Tabla de contenidos

1. [Arquitectura general](#1-arquitectura-general)
2. [Flujo de señal paso a paso](#2-flujo-de-señal-paso-a-paso)
3. [Sistema multi-estrategia v2](#3-sistema-multi-estrategia-v2)
4. [Capas de riesgo](#4-capas-de-riesgo)
5. [Base de datos](#5-base-de-datos)
6. [Variables de entorno críticas](#6-variables-de-entorno-críticas)
7. [Instalación y ejecución](#7-instalación-y-ejecución)
8. [Backtesting](#8-backtesting)
9. [Tests](#9-tests)
10. [Monitoreo y observabilidad](#10-monitoreo-y-observabilidad)
11. [Estructura del repositorio](#11-estructura-del-repositorio)

---

## 1. Arquitectura general

El sistema opera en **dos procesos independientes**. Nunca deben combinarse:

| Proceso | Archivo | Binance endpoint | Función |
|---|---|---|---|
| **Market Data** | `market_data_process.py` | MAINNET (datos) | Ingesta OHLCV → PostgreSQL |
| **Trading Bot** | `bot_trading_v5.py` | TESTNET (órdenes) | Lee DB → decide → ejecuta |

PostgreSQL es la **fuente única de verdad**. El bot nunca lee klines de Binance directamente;
todo el histórico viene de `trading.market_klines`, rellenado por el proceso de datos.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PROCESO DE DATOS                            │
│  market_data_process.py                                             │
│  • Binance MAINNET /klines                                          │
│  • Escribe en trading.market_klines (PostgreSQL)                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ PostgreSQL
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          TRADING BOT                                │
│  bot_trading_v5.py — hilos por símbolo                             │
│                                                                     │
│  ┌──────────────┐  ┌───────────────────┐  ┌────────────────────┐   │
│  │ run_strategy │  │ monitoring_open_  │  │  Workers globales  │   │
│  │ (señales +   │  │ position()        │  │  reconcile, equity │   │
│  │  BUY)        │  │ (SL/TP/trailing)  │  │  métricas, health  │   │
│  └──────┬───────┘  └────────┬──────────┘  └────────────────────┘   │
│         │                   │                                        │
│  StrategyEngine        Binance TESTNET                              │
│  PortfolioManager      (solo órdenes)                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Principios de diseño

- **Sin CSV**: `USE_DATABASE=true` es obligatorio. Toda la persistencia es PostgreSQL.
- **Fail-open en capas secundarias**: Risk v3, MultiEngine y StrategyEngine nunca bloquean el bot si fallan al inicializar. Solo el modelo ML es fail-closed para entradas.
- **Sin lookahead**: todas las señales se calculan sobre `iloc[-2]` (última vela cerrada).
- **Multi-hilo**: un par de hilos `run_strategy` + `monitoring_open_position` por símbolo, más workers globales compartidos.
- **TESTNET para órdenes**: `client_trade` (TESTNET) ejecuta todas las órdenes. `client_data` (MAINNET) solo consulta filtros de símbolo.

---

## 2. Flujo de señal paso a paso

Cada `POLL_INTERVAL_SECONDS` (default: 30s) el hilo `run_strategy(symbol)` ejecuta:

```
1. get_data_binance(symbol)
   └─ Lee trading.market_klines desde PostgreSQL (últimas 260 velas)

2. _add_cross_asset_features()
   └─ Añade columnas cross-asset (btc_dominance, eth_correlation_30)
      si el modelo ML las requiere

3. strategy.prepare_indicators(df)
   └─ Calcula: EMA50, EMA200, ADX, ATR, RSI, BB, StochRSI
   └─ Detecta régimen por fila (BULL / BEAR / LATERAL) sin lookahead

4. StrategyEngine.collect(MarketState)
   └─ Evalúa las 5 estrategias en paralelo (fail-safe por estrategia)
   └─ Retorna list[Signal]  (BUY | HOLD | SELL)

5. PortfolioManager.decide(signals)
   └─ Scoring ponderado: score = Σ(weight × confidence)
   └─ veto_on_hold=True → cualquier HOLD bloquea la entrada
   └─ Retorna PortfolioOrder (side, size_frac, triggered_by)

6. ¿entry_ok?
   ├─ NO → esperar siguiente ciclo
   └─ SÍ → Risk checks (v2 + v3) → ejecutar_orden_con_confirmacion('buy')
             └─ Guardar posición en trading.open_positions
             └─ Notificar Telegram

7. DCA (si ENABLE_DCA=true)
   └─ Si mercado cae ≥ 5% en últimas 5 velas y no hay posición abierta:
      compra adicional al 50% del position_size normal
      (nunca en régimen BEAR)
```

### Gestión de posiciones abiertas (`monitoring_open_position`)

El hilo monitor evalúa cada posición abierta en cada ciclo:

```
Para cada posición:
  1. Verificar Take-Profit fijo (TP = buy_price + 2.5 × ATR)
  2. Activar trailing stop cuando close >= tp_initial
  3. Si trailing activo: actualizar stop_loss = max_price − 1.2 × ATR
  4. Verificar Stop-Loss (close <= stop_loss) → SELL
  5. Si BEAR y posición flotante con pérdida → SELL preventivo
```

---

## 3. Sistema multi-estrategia v2

### Estrategias disponibles

| Estrategia | `strategy_id` | Peso | Símbolos | Lógica |
|---|---|---|---|---|
| **MLStrategy** | `ml_momentum` | 2.0 | Todos | Modelo gradient boosting + EMA/ADX |
| **TrendStrategy** | `trend_following` | 1.5 | BTC, ETH | EMA50 > EMA200, ADX ≥ 25, RSI [35–65] |
| **MeanReversionStrategy** | `mean_reversion` | 1.0 | Altcoins | BB inferior + RSI cruzando desde sobreventa |
| **FundingArbitrageStrategy** | `funding_arb` | 0.5 | BTC, ETH | Filtro macro: bloquea si funding > threshold |
| **VolatilityBreakoutStrategy** | `vol_breakout` | 1.0 | Todos | ATR en expansión + rotura de BB |

### Lógica de consenso (PortfolioManager)

```
score_buy  = Σ(weight × confidence)  de señales BUY
score_sell = Σ(weight × confidence)  de señales SELL

Si veto_on_hold=true y hay algún HOLD → HOLD (no entrar)
Si score_buy  ≥ buy_threshold  y score_buy  > score_sell → BUY
Si score_sell ≥ sell_threshold y score_sell > score_buy  → SELL
En caso contrario → HOLD

size_frac = min(size_frac de señales BUY activas)  ← siempre conservador
```

Con la configuración por defecto (`buy_threshold=1.0`, `veto_on_hold=true`):
- El **MLStrategy** (peso 2.0, confianza ≈ ml_prob) puede aprobar la entrada solo.
- Un **HOLD** de funding_arb (funding excesivo) **veta** la entrada aunque ML diga BUY.
- El tamaño de posición es el **mínimo** de todas las estrategias que aprueban.

### Diagrama de módulos

```
strategies/
├── base_strategy.py        → BaseStrategy, Signal, MarketState
├── ml_strategy.py          → MLStrategy (wraps BotV5StrategyAdapter)
├── trend_strategy.py       → TrendStrategy
├── mean_reversion_strategy.py
├── funding_arbitrage_strategy.py
└── volatility_breakout_strategy.py

strategy_engine.py          → StrategyEngine.collect(state) → list[Signal]
portfolio_manager.py        → PortfolioManager.decide(symbol, signals) → PortfolioOrder
estrategia_v5.py            → BotV5StrategyAdapter (backtesting + ML)
estrategia_multi.py         → MultiStrategyBacktestAdapter (backtesting multi-estrategia)
strategies_multi.py         → MultiStrategyEngine legacy (4 estrategias, modo ANY/MAJORITY/ALL)
```

---

## 4. Capas de riesgo

### Risk Layer v2 (siempre activo)

| Componente | Función |
|---|---|
| `GlobalRiskController` | Kill switch si drawdown supera `GLOBAL_MAX_DRAWDOWN_FRAC` |
| `RiskEventLogger` | Auditoría: escribe eventos en `trading.risk_events` |
| `SystemHealthMonitor` | Circuit breaker tras `MAX_CRITICAL_ERRORS` errores consecutivos |
| `reconcile_worker` | Sincroniza posiciones DB ↔ Binance cada ciclo |
| `risk_metrics_worker` | Escribe métricas diarias en `trading.risk_metrics_daily` |
| `start_health_server` | Endpoint HTTP `GET /health` (Flask) |

También opera el **RiskManager por símbolo** (en memoria):

| Control | Variable | Default |
|---|---|---|
| Drawdown máximo por símbolo | `MAX_SYMBOL_DRAWDOWN_FRAC` | 6% |
| Capital comprometido máximo | `MAX_CAPITAL_COMMITTED_FRAC` | 15% del equity |
| Pérdida diaria límite | `DAILY_LOSS_LIMIT_FRAC` | 3% del equity diario |
| Cooldown tras pérdida diaria | `SYMBOL_COOLDOWN_AFTER_DAILY_LOSS_SECONDS` | 86400s (24h) |
| Cooldown entre compras | `BUY_COOLDOWN_SECONDS` | 5400s (90min) |

### Risk Layer v3 (opcional, fail-open)

Cada componente se activa de forma independiente. Si falla en init, el bot continúa sin él.

| Componente | Variable de activación | Función |
|---|---|---|
| `VolatilityPositionSizer` | `RISK_V3_POSITION_SIZER_ENABLED=true` | Ajusta tamaño por ATR |
| `PortfolioCorrelationRisk` | `RISK_V3_CORRELATION_ENABLED=true` | Limita exposición correlacionada |
| `IntradayVaRMonitor` | `RISK_V3_VAR_ENABLED=true` | Bloquea si VaR > límite |
| `SlippageMonitor` | `RISK_V3_SLIPPAGE_ENABLED=true` | Para bot si slippage excesivo |
| `EquityRegimeFilter` | `RISK_V3_EQUITY_REGIME_ENABLED=true` | Reduce/bloquea en tendencia de equity negativa |

---

## 5. Base de datos

### Schema `trading` (PostgreSQL)

| Tabla | Propósito |
|---|---|
| `trading.market_klines` | OHLCV histórico (PK: symbol + interval + open_time) |
| `trading.open_positions` | Posiciones activas con TP/SL/trailing |
| `trading.trade_history` | Registro de todos los trades ejecutados |
| `trading.equity_snapshots` | Serie temporal de equity total |
| `trading.global_risk_state` | Estado del kill switch global |
| `trading.risk_events` | Auditoría de eventos de riesgo |
| `trading.reconciliation_events` | Log de reconciliaciones DB ↔ Binance |
| `trading.risk_metrics_daily` | Métricas diarias de riesgo |

El schema es **idempotente**: el bot lo crea con `CREATE TABLE IF NOT EXISTS` al arrancar.
Para aplicarlo manualmente: `psql -d trading -f db/schema.sql`

---

## 6. Variables de entorno críticas

Copiar `.env.example` a `.env` y completar. Las marcadas con ⚠️ son **obligatorias**.

### 6.1 Binance

```dotenv
# ⚠️ Credenciales TESTNET (solo ejecución de órdenes)
BINANCE_TRADE_API_KEY=<tu_testnet_api_key>
BINANCE_TRADE_API_SECRET=<tu_testnet_api_secret>

# Credenciales MAINNET (datos de mercado — opcional si usas endpoints públicos)
BINANCE_DATA_API_KEY=
BINANCE_DATA_API_SECRET=

BINANCE_TESTNET=true     # Siempre true para trading real en testnet
```

> **Importante**: `BINANCE_TRADE_*` es para el cliente de ejecución (TESTNET).
> `BINANCE_DATA_*` es para el cliente de datos (MAINNET). Son clientes separados.

### 6.2 PostgreSQL

```dotenv
# ⚠️ Obligatorio — el bot aborta si USE_DATABASE=false
USE_DATABASE=true

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading
POSTGRES_USER=trader
POSTGRES_PASSWORD=changeme

# Tuning de pool (opcional)
POSTGRES_POOL_MIN=1
POSTGRES_POOL_MAX=8
POSTGRES_CONNECT_TIMEOUT=5
```

### 6.3 Estrategia principal

```dotenv
# ⚠️ Símbolos a operar (separados por coma, mayúsculas)
SYMBOLS=BTCUSDT,ETHUSDT

# ⚠️ Timeframe de velas (debe coincidir con lo que sincroniza market_data_process)
TIMEFRAME=1h

# Intervalo de polling del bot (segundos)
POLL_INTERVAL_SECONDS=30

# Cooldown entre compras por símbolo (segundos)
BUY_COOLDOWN_SECONDS=5400
```

### 6.4 Modelo ML

```dotenv
# ⚠️ Ruta al modelo — el bot NO entra si no puede cargarlo
ML_MODEL_PATH=artifacts/model_momentum_v4_phase2.joblib

# Umbral de probabilidad para señal de entrada (0.0–1.0)
ML_PROB_THRESHOLD=0.60

# Override de ruta/umbral sin tocar el código (útil en CI/staging)
V5_MODEL_PATH=
V5_THRESHOLD_OVERRIDE=
```

> El modelo es **fail-closed**: si no carga o faltan features, el bot **no entra**.

### 6.5 Sistema multi-estrategia v2

```dotenv
# Threshold de score ponderado para aprobar BUY (suma weight×confidence)
PORTFOLIO_BUY_THRESHOLD=1.0

# Threshold para SELL
PORTFOLIO_SELL_THRESHOLD=1.0

# true = cualquier señal HOLD veta la entrada (recomendado)
# false = solo cuenta el score neto
PORTFOLIO_VETO_ON_HOLD=true

# --- TrendStrategy ---
TREND_SYMBOLS=BTCUSDT,ETHUSDT
TREND_EMA_FAST=50
TREND_EMA_SLOW=200
TREND_ADX_MIN=25
TREND_RSI_MIN=35
TREND_RSI_MAX=65
TREND_POSITION_SIZE=0.08

# --- MeanReversionStrategy ---
MEAN_REV_SYMBOLS=RE:^(?!BTC|ETH).+USDT$    # regex: altcoins USDT
MEAN_REV_BB_WINDOW=20
MEAN_REV_BB_STD=2.0
MEAN_REV_RSI_OS=30
MEAN_REV_RSI_OB=70
MEAN_REV_POSITION_SIZE=0.04

# --- FundingArbitrageStrategy (filtro macro) ---
# Bloquea entrada si funding rate > umbral (mercado sobrecalentado)
FUNDING_SYMBOLS=BTCUSDT,ETHUSDT
FUNDING_THRESHOLD=0.0005    # 0.05%

# --- VolatilityBreakoutStrategy ---
VOL_BREAKOUT_ATR_MULT=1.5
VOL_BREAKOUT_LOOKBACK=20
VOL_BREAKOUT_BB_LAG=3
```

### 6.6 Risk Layer v2

```dotenv
# Kill switch global (drawdown desde peak de equity)
GLOBAL_KILL_SWITCH_ENABLED=true
GLOBAL_MAX_DRAWDOWN_FRAC=0.10      # 10% → para todo el bot

# Circuit breaker
MAX_CRITICAL_ERRORS=5

# Risk por símbolo (en memoria, reset al reiniciar)
MAX_SYMBOL_DRAWDOWN_FRAC=0.06
MAX_CAPITAL_COMMITTED_FRAC=0.15
DAILY_LOSS_LIMIT_FRAC=0.03
SYMBOL_COOLDOWN_AFTER_DAILY_LOSS_SECONDS=86400
BUY_COOLDOWN_SECONDS=5400

# Configuración de posiciones recuperadas (reconcile)
RECOVERED_ATR_INTERVAL=1h
RECOVERED_ATR_PERIOD=14
RECOVERED_SL_ATR_MULT=2.0
RECOVERED_TP_ATR_MULT=2.0
RECOVERED_TRAILING_SL_ATR_MULT=2.0

# Cancelar órdenes abiertas al intentar vender si hay fondos bloqueados
CANCEL_OPEN_ORDERS_ON_STOP=1
```

### 6.7 Risk Layer v3 (opcional)

```dotenv
# 1) Position Sizer adaptado a volatilidad
RISK_V3_POSITION_SIZER_ENABLED=false
RISK_PER_TRADE_FRAC=0.01           # Riesgo por trade = 1% del equity
MAX_POSITION_CAP_FRAC=0.10         # Máximo 10% del equity por posición
MIN_POSITION_USDT=10.0
POSITION_SIZER_ATR_MULTIPLIER=1.0
POSITION_SIZER_ATR_WINDOW=14

# 2) Correlación entre símbolos
RISK_V3_CORRELATION_ENABLED=false
CORRELATION_WINDOW_DAYS=30
CORRELATION_THRESHOLD=0.8
CORRELATION_MAX_COMBINED_EXPOSURE_FRAC=0.25

# 3) VaR histórico intradiario
RISK_V3_VAR_ENABLED=false
VAR_WINDOW_DAYS=30
VAR_CONFIDENCE=0.95
MAX_VAR_FRAC=0.05                  # Para nuevas entradas si VaR > 5%

# 4) Slippage monitor
RISK_V3_SLIPPAGE_ENABLED=false
SLIPPAGE_MAX_FRAC=0.005            # 0.5% max slippage aceptable
SLIPPAGE_MAX_CONSECUTIVE=3         # N consecutivos → detiene el bot

# 5) Equity regime filter
RISK_V3_EQUITY_REGIME_ENABLED=false
EQUITY_REGIME_MODE=reduce          # 'reduce' o 'block'
EQUITY_REGIME_REDUCTION_FRAC=0.5
EQUITY_EMA_FAST=50
EQUITY_EMA_SLOW=200
```

### 6.8 ATR y gestión de posición

```dotenv
# Ventana ATR para TP/SL/trailing (velas)
ATR_WINDOW=14

# TP = buy_price + 2.5 × ATR   (fijo en estrategia_v5)
# SL = buy_price − 1.5 × ATR
# Trailing SL se activa al alcanzar TP y se mueve: max_price − 1.2 × ATR

# DCA: compra adicional ante caídas (desactivado por defecto)
ENABLE_DCA=0                        # 1 para activar, 0 para desactivar
```

### 6.9 Market Data Process

```dotenv
# Símbolos e intervalos para el proceso de ingesta (separados por coma)
MARKET_DATA_SYMBOLS=BTCUSDT,ETHUSDT
MARKET_DATA_INTERVALS=1h

# Frecuencia de sincronización con Binance (segundos)
MARKET_DATA_SYNC_EVERY_SECONDS=30

# Backfill inicial (máx velas por llamada a Binance)
KLINES_BACKFILL_LIMIT=1000
KLINES_MAX_PAGES=20
KLINES_OVERLAP_CANDLES=2

# Rango histórico específico (opcional)
KLINES_START_TIME=2024-01-01T00:00:00Z
KLINES_END_TIME=
```

### 6.10 Infraestructura

```dotenv
# Health server (requiere Flask)
HEALTH_HOST=0.0.0.0
HEALTH_PORT=8000

# Frecuencia de snapshots de equity (segundos)
EQUITY_SNAPSHOT_INTERVAL=300

# Telegram (opcional pero muy recomendado en producción)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Nivel de logging
LOG_LEVEL=INFO                      # DEBUG para diagnóstico detallado
```

---

## 7. Instalación y ejecución

### Requisitos

- Python 3.12+
- PostgreSQL 14+
- Cuenta de Binance con API activada en TESTNET

### Paso a paso

```bash
# 1. Clonar y configurar entorno
git clone <repo-url>
cd bot-crypto-trading-BNB
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales (ver sección 6)

# 3. Crear el schema de PostgreSQL (idempotente)
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f db/schema.sql

# 4. [Terminal 1] Arrancar el proceso de datos (PRIMERO)
source .venv/bin/activate
python market_data_process.py
# Esperar al menos 260 velas sincronizadas antes de arrancar el bot

# 5. [Terminal 2] Arrancar el bot de trading
source .venv/bin/activate
python bot_trading_v5.py

# 6. Verificar estado (en otra terminal)
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Verificar que hay datos suficientes antes de arrancar el bot

```sql
-- Contar velas disponibles por símbolo/intervalo
SELECT symbol, interval, COUNT(*) AS n_candles,
       MIN(open_time) AS desde, MAX(open_time) AS hasta
FROM trading.market_klines
GROUP BY 1, 2
ORDER BY 1, 2;
-- Necesitas al menos 260 velas para que EMA200 esté disponible
```

### Docker (opcional)

```bash
# Levantar PostgreSQL y ambos procesos
docker compose up -d

# Ver logs del bot
docker compose logs -f bot

# Ver logs del proceso de datos
docker compose logs -f market_data
```

---

## 8. Backtesting

El framework de backtesting simula el bot vela por vela sobre datos históricos,
usando exactamente la misma lógica de estrategia que en live (sin lookahead).

### Con la estrategia ML (BotV5StrategyAdapter)

```bash
# Backtest estándar con modelo ML
python -m backtesting.run_backtest \
  --symbol BTCUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --strategy estrategia_v5.BotV5StrategyAdapter \
  --output backtest_out/

# Ver ayuda completa
python -m backtesting.run_backtest --help
```

### Con el sistema multi-estrategia v2

```python
# En un script Python:
from backtesting.run_backtest import run_backtest
from estrategia_multi import MultiStrategyBacktestAdapter

run_backtest(
    symbol="BTCUSDT",
    interval="1h",
    start="2024-01-01",
    end="2025-01-01",
    strategy=MultiStrategyBacktestAdapter(),
    output_dir="backtest_out/",
)
```

### Outputs

| Archivo | Contenido |
|---|---|
| `backtest_out/equity_curve.csv` | Equity total por vela |
| `backtest_out/trades.csv` | Detalle de cada trade (entry/exit/PnL) |

---

## 9. Tests

```bash
# Activar entorno
source .venv/bin/activate

# Ejecutar todos los tests
pytest tests/ -v

# Solo tests rápidos (sin DB ni red)
pytest tests/test_strategies_multi.py \
       tests/test_strategy_engine_and_portfolio.py \
       tests/test_risk_layer_v2.py \
       tests/test_risk_layer_v3.py -v

# Tests de integración del bot
pytest tests/test_bot_trading_v5_integration.py -v
```

### Cobertura de tests

| Archivo de test | Qué cubre | Tests |
|---|---|---|
| `test_strategies_multi.py` | 4 estrategias legacy + MultiStrategyEngine | 35 |
| `test_strategy_engine_and_portfolio.py` | StrategyEngine, PortfolioManager, Signal, MarketState | 27 |
| `test_bot_trading_v5_integration.py` | `_build_v5_strategy_adapter`, `run_strategy` end-to-end | 2 |
| `test_risk_layer_v2.py` | GlobalRisk, HealthMonitor, reconcile | — |
| `test_risk_layer_v3.py` | VaR, correlación, slippage, sizing, regime | — |
| `test_estrategia_v5.py` | BotV5StrategyAdapter, prepare_indicators, generate_entry | — |
| `test_v3_1_architecture_and_regression.py` | Regresión arquitectura v3.1 | — |

---

## 10. Monitoreo y observabilidad

### Health endpoint

```bash
curl http://localhost:8000/health
```

Respuesta típica:
```json
{
  "status": "ok",
  "kill_switch": false,
  "drawdown_frac": 0.012,
  "peak_equity": 1050.50,
  "current_equity": 1037.88,
  "critical_errors": 0
}
```

### Logs

Los logs se escriben en `./logs/bot_trading.log` (rotativo, 10 MB × 5 archivos)
y también a stdout para integración con Docker.

```bash
# Seguir logs en tiempo real
tail -f logs/bot_trading.log

# Filtrar por símbolo
grep "\[BTCUSDT\]" logs/bot_trading.log | tail -50

# Ver solo señales de entrada
grep "Señal entrada" logs/bot_trading.log
```

### Telegram

Si configuras `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`, el bot envía notificaciones de:
- COMPRA ejecutada
- VENTA ejecutada (con motivo: SL / TP / trailing / BEAR)
- Alertas de kill switch o circuit breaker

Comandos disponibles via Telegram:
- `/status` — estado del bot y posiciones abiertas
- `/balance` — balance actual en TESTNET
- `/stop` — detener el bot de forma controlada

### Queries útiles en PostgreSQL

```sql
-- Posiciones actualmente abiertas
SELECT symbol, buy_price, amount, take_profit, stop_loss,
       trailing_active, regime, opened_at
FROM trading.open_positions
ORDER BY opened_at DESC;

-- Últimos 10 trades
SELECT symbol, side, reason, buy_price, sell_price,
       realized_pnl, realized_pnl_pct, regime, executed_at
FROM trading.trade_history
ORDER BY executed_at DESC
LIMIT 10;

-- Evolución de equity (últimas 24h)
SELECT timestamp, equity_total, usdt_balance, positions_value
FROM trading.equity_snapshots
WHERE timestamp > now() - INTERVAL '24 hours'
ORDER BY timestamp;

-- PnL realizado por símbolo
SELECT symbol,
       COUNT(*) FILTER (WHERE side='sell') AS trades,
       SUM(realized_pnl) AS pnl_total,
       AVG(realized_pnl_pct) AS avg_pnl_pct
FROM trading.trade_history
WHERE side = 'sell'
GROUP BY 1
ORDER BY pnl_total DESC;

-- Eventos de riesgo recientes
SELECT event_type, severity, message, created_at
FROM trading.risk_events
ORDER BY created_at DESC
LIMIT 20;
```

---

## 11. Estructura del repositorio

```
bot-crypto-trading-BNB/
│
├── bot_trading_v5.py              # Entrypoint del bot (orquestación principal)
├── market_data_process.py         # Proceso de ingesta OHLCV → PostgreSQL
│
├── strategies/                    # Sistema multi-estrategia v2
│   ├── __init__.py
│   ├── base_strategy.py           # BaseStrategy, Signal, MarketState
│   ├── ml_strategy.py             # MLStrategy (wraps BotV5StrategyAdapter)
│   ├── trend_strategy.py          # TrendStrategy (BTC/ETH)
│   ├── mean_reversion_strategy.py # MeanReversionStrategy (altcoins)
│   ├── funding_arbitrage_strategy.py  # FundingArbitrageStrategy (filtro)
│   └── volatility_breakout_strategy.py
│
├── strategy_engine.py             # StrategyEngine.collect() → list[Signal]
├── portfolio_manager.py           # PortfolioManager.decide() → PortfolioOrder
├── strategies_multi.py            # MultiStrategyEngine legacy (ANY/MAJORITY/ALL)
│
├── estrategia_v5.py               # BotV5StrategyAdapter (ML + indicadores)
├── estrategia_multi.py            # Adaptador de backtesting multi-estrategia
│
├── risk_layer_v2.py               # Risk institucional v2 (siempre activo)
├── risk_layer_v3.py               # Risk avanzado v3 (modular, fail-open)
│
├── db/
│   ├── connection.py              # Pool de conexiones PostgreSQL
│   ├── schema.py                  # Creación idempotente de tablas
│   └── schema.sql                 # SQL puro (aplicación manual)
│
├── repositories/
│   ├── open_positions_repo.py     # CRUD posiciones abiertas
│   ├── trade_history_repo.py      # Inserción de trades
│   └── equity_repo.py             # Snapshots de equity
│
├── services/
│   └── market_klines_service.py   # Lectura de klines desde PostgreSQL
│
├── backtesting/
│   ├── bt_types.py                # Strategy, StrategyContext, EntrySignal, Position, Bar
│   └── run_backtest.py            # Runner vela-por-vela (sin lookahead)
│
├── artifacts/
│   └── model_momentum_v4_phase2.joblib  # Modelo ML (gradient boosting)
│
├── tests/                         # Suite de tests unitarios e integración
│
├── tools/
│   ├── env_audit.py               # Auditoría de variables de entorno
│   └── kline_gap_audit.py         # Detección de gaps en market_klines
│
├── .env.example                   # Plantilla de variables de entorno
├── requirements.txt
└── logs/
    └── bot_trading.log            # Rotativo 10 MB × 5 archivos
```
