
# Bot Crypto Trading v5 — Binance Spot (Multi-Strategy)

Sistema de trading algorítmico para Binance Spot con arquitectura multi-estrategia,
ML híbrido como escalador de posición, capas de riesgo institucional y persistencia
total en PostgreSQL. Desplegable en contenedores Docker independientes.

---

## Tabla de contenidos

1. [Arquitectura general](#1-arquitectura-general)
2. [Flujo de señal paso a paso](#2-flujo-de-señal-paso-a-paso)
3. [Sistema multi-estrategia v2](#3-sistema-multi-estrategia-v2)
4. [ML Modo Híbrido](#4-ml-modo-híbrido)
5. [Capas de riesgo](#5-capas-de-riesgo)
6. [Base de datos](#6-base-de-datos)
7. [Variables de entorno críticas](#7-variables-de-entorno-críticas)
8. [Instalación y ejecución](#8-instalación-y-ejecución)
9. [Despliegue con Docker](#9-despliegue-con-docker)
10. [Backtesting](#10-backtesting)
11. [Tests](#11-tests)
12. [Monitoreo y observabilidad](#12-monitoreo-y-observabilidad)
13. [Estructura del repositorio](#13-estructura-del-repositorio)

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
   └─ MLStrategy incluye raw_prob en Signal.confidence aunque sea HOLD

5. PortfolioManager.decide(signals)   ← Modo Híbrido (ver sección 4)
   └─ Extrae ml_conf del MLStrategy signal
   └─ Si ml_conf < ML_MIN_CONFIDENCE → HOLD (gate mínimo)
   └─ Score ponderado sin ML: score = Σ(weight × confidence) de Trend + Vol
   └─ Si score ≥ buy_threshold → BUY con size_frac × f(ml_conf)
   └─ Retorna PortfolioOrder (side, size_frac escalado, triggered_by)

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
| **MLStrategy** | `ml_momentum` | 2.0 | Todos | Gradient boosting 17 features; en modo híbrido actúa como escalador |
| **TrendStrategy** | `trend_following` | 1.5 | BTC, ETH | EMA50 > EMA200, ADX ≥ 25, RSI [35–65] |
| **MeanReversionStrategy** | `mean_reversion` | 1.0 | Altcoins | BB inferior + RSI cruzando desde sobreventa |
| **FundingArbitrageStrategy** | `funding_arb` | 0.5 | BTC, ETH | Bloquea si funding rate > threshold |
| **VolatilityBreakoutStrategy** | `vol_breakout` | 1.0 | Todos | ATR en expansión + rotura de BB |

### Lógica de consenso (modo estándar)

```
score_buy  = Σ(weight × confidence)  de señales BUY
score_sell = Σ(weight × confidence)  de señales SELL

Si veto_on_hold=true y hay algún HOLD → HOLD (no entrar)
Si score_buy  ≥ buy_threshold  y score_buy  > score_sell → BUY
Si score_sell ≥ sell_threshold y score_sell > score_buy  → SELL
En caso contrario → HOLD

size_frac = min(size_frac de señales BUY activas)  ← siempre conservador
```

> En producción se usa el **Modo Híbrido** (ver sección 4), que reemplaza el veto
> binario por un escalado continuo de tamaño.

### Diagrama de módulos

```
strategies/
├── base_strategy.py        → BaseStrategy, Signal, MarketState
├── ml_strategy.py          → MLStrategy (wraps BotV5StrategyAdapter)
│                              HOLD ahora lleva raw_prob en confidence
├── trend_strategy.py       → TrendStrategy
├── mean_reversion_strategy.py
├── funding_arbitrage_strategy.py
└── volatility_breakout_strategy.py

strategy_engine.py          → StrategyEngine.collect(state) → list[Signal]
portfolio_manager.py        → PortfolioManager.decide(symbol, signals) → PortfolioOrder
                               Soporta modo estándar (veto binario) y modo híbrido (escalado)
estrategia_v5.py            → BotV5StrategyAdapter (backtesting + ML)
estrategia_multi.py         → MultiStrategyBacktestAdapter (backtesting multi-estrategia)
                               Fix: almacena df completo por símbolo para evitar bug iloc[-2]
strategies_multi.py         → MultiStrategyEngine legacy (4 estrategias, modo ANY/MAJORITY/ALL)
```

---

## 4. ML Modo Híbrido

### Problema del filtro binario

Con `PORTFOLIO_VETO_ON_HOLD=true` y `ML_PROB_THRESHOLD=0.60`, el modelo ML actúa
como **veto binario**: cualquier HOLD del ML (prob < 0.60) bloquea la entrada aunque
Trend y Vol estén de acuerdo. Backtest sobre 3 años confirmó que esto **paraliza el bot**:

| Año | BTC Buy&Hold | ML Filtro binario | Trend libre |
|---|---|---|---|
| 2022 (Bear) | -65% | 0 trades, 0% | 174 trades, -4.67% |
| 2024 (Bull) | +117.7% | 0 trades, 0% | 232 trades, -0.43% |
| 2025 (Volátil) | -7.2% | 0 trades, 0% | 233 trades, -5.02% |

### Solución: ML como escalador de posición

El ML deja de ser un gate binario y pasa a **escalar el tamaño** de la posición
según su nivel de confianza. El trigger de entrada lo mantienen Trend y VolBreakout.

```
Confianza ML    Acción             Tamaño
──────────────  ──────────────     ──────────────────────
< ML_MIN_CONF   No entrar (gate)   0%
[min, 0.55)     Entrada mínima     base_size × ML_SIZE_SCALE_LOW  (×0.5)
[0.55, 0.70)    Entrada normal     base_size × ML_SIZE_SCALE_MID  (×1.0)
≥ 0.70          Entrada ampliada   base_size × ML_SIZE_SCALE_HIGH (×1.5)
```

### Resultados backtest 2022–2025 (BTCUSDT 1h, $10k, fee 0.1%)

| Año | BTC B&H | Trend Libre | **ML Híbrido** | Mejora DD |
|---|---|---|---|---|
| 2022 (Bear) | -65% | -4.67% / DD 5.40% | **-2.26% / DD 2.77%** | -49% |
| 2024 (Bull) | +117% | -0.43% / DD 3.35% | **-0.35% / DD 2.05%** | -39% |
| 2025 (Volátil) | -7.2% | -5.02% / DD 5.27% | **-1.88% / DD 2.45%** | -53% |

El modo híbrido **reduce el drawdown máximo entre 39-53% en todos los regímenes**
sin paralizar el bot como hacía el filtro binario.

### Implementación técnica

- `strategies/ml_strategy.py`: cuando ML devuelve HOLD, incluye `raw_prob` en
  `Signal.confidence` (antes era siempre 0.0). Esto permite al PortfolioManager
  aplicar el gate mínimo y escalar sin necesidad de una señal BUY explícita del ML.
- `portfolio_manager.py`: método `_ml_size_scale(ml_conf)` devuelve el factor de
  escala. En modo híbrido, el score de entrada ignora el ML y usa solo las
  estrategias técnicas (Trend + Vol).

### Activar / desactivar

```dotenv
# Modo híbrido (recomendado en producción)
ML_HYBRID_MODE=true
PORTFOLIO_VETO_ON_HOLD=false

# Modo binario clásico (desactiva el híbrido)
ML_HYBRID_MODE=false
PORTFOLIO_VETO_ON_HOLD=true
```

---

## 5. Capas de riesgo

### Risk Layer v2 (siempre activo)

| Componente | Función |
|---|---|
| `GlobalRiskController` | Kill switch si drawdown supera `GLOBAL_MAX_DRAWDOWN_FRAC` |
| `RiskEventLogger` | Auditoría: escribe eventos en `trading.risk_events` |
| `SystemHealthMonitor` | Circuit breaker tras `MAX_CRITICAL_ERRORS` errores consecutivos |
| `reconcile_worker` | Sincroniza posiciones DB ↔ Binance cada ciclo |
| `risk_metrics_worker` | Escribe métricas diarias en `trading.risk_metrics_daily` |
| `start_health_server` | Endpoint HTTP `GET /health` (Flask, puerto `HEALTH_PORT`) |

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

## 6. Base de datos

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
Para aplicarlo manualmente: `psql -d bot_trading -f db/schema.sql`

---

## 7. Variables de entorno críticas

Copiar `.env.example` a `.env` y completar. Las marcadas con ⚠️ son **obligatorias**.

### 7.1 Binance

```dotenv
# ⚠️ Credenciales TESTNET (solo ejecución de órdenes)
BINANCE_TRADE_API_KEY=<tu_testnet_api_key>
BINANCE_TRADE_API_SECRET=<tu_testnet_api_secret>

# Credenciales MAINNET (datos de mercado — opcional si usas endpoints públicos)
BINANCE_DATA_API_KEY=
BINANCE_DATA_API_SECRET=

BINANCE_TESTNET=true     # Siempre true para trading en testnet
```

> **Importante**: `BINANCE_TRADE_*` es para el cliente de ejecución (TESTNET).
> `BINANCE_DATA_*` es para el cliente de datos (MAINNET). Son clientes separados.

### 7.2 PostgreSQL

```dotenv
# ⚠️ Obligatorio — el bot aborta si USE_DATABASE=false
USE_DATABASE=true

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=bot_trading
POSTGRES_USER=postgres
POSTGRES_PASSWORD=changeme

# Tuning de pool (opcional)
POSTGRES_POOL_MIN=1
POSTGRES_POOL_MAX=8
POSTGRES_CONNECT_TIMEOUT=5
```

### 7.3 Estrategia principal

```dotenv
# ⚠️ Símbolos a operar (separados por coma, mayúsculas)
SYMBOLS=BTCUSDT

# ⚠️ Timeframe de velas (debe coincidir con lo que sincroniza market_data_process)
TIMEFRAME=1h

POLL_INTERVAL_SECONDS=30
BUY_COOLDOWN_SECONDS=5400
```

### 7.4 Modelo ML

```dotenv
# ⚠️ Ruta al modelo — el bot NO entra si no puede cargarlo
ML_MODEL_PATH=artifacts/model_momentum_v4_phase2.joblib

# Umbral de probabilidad para señal BUY del ML (0.0–1.0)
ML_PROB_THRESHOLD=0.60
```

> El modelo es **fail-closed**: si no carga o faltan features, el bot **no entra**.
> Usa 17 features: indicadores técnicos (EMA, ADX, ATR, RSI, MACD, BB) +
> cross-asset (eth_correlation_30, btc_dominance, market_breadth).

### 7.5 Sistema multi-estrategia v2 + ML Híbrido

```dotenv
# Score mínimo ponderado para aprobar BUY/SELL
PORTFOLIO_BUY_THRESHOLD=1.0
PORTFOLIO_SELL_THRESHOLD=1.0

# ── ML Modo Híbrido (recomendado) ────────────────────────────────────
# El ML escala el tamaño de posición en vez de actuar como veto binario.
# Backtest 2022-2025: reduce drawdown 39-53% vs filtro binario.
PORTFOLIO_VETO_ON_HOLD=false      # false = compatible con modo híbrido
ML_HYBRID_MODE=true               # activar escalado por confianza ML
ML_HYBRID_STRATEGY_ID=ml_momentum
ML_MIN_CONFIDENCE=0.40            # gate: conf < 0.40 → no entra
ML_SIZE_SCALE_LOW=0.5             # conf [0.40, 0.55) → base_size × 0.5
ML_SIZE_SCALE_MID=1.0             # conf [0.55, 0.70) → base_size × 1.0
ML_SIZE_SCALE_HIGH=1.5            # conf ≥ 0.70       → base_size × 1.5

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
MEAN_REV_RSI_OS=30
MEAN_REV_POSITION_SIZE=0.04

# --- FundingArbitrageStrategy (filtro macro) ---
FUNDING_SYMBOLS=BTCUSDT,ETHUSDT
FUNDING_THRESHOLD=0.0005    # 0.05%
FUNDING_POSITION_SIZE=0.05

# --- VolatilityBreakoutStrategy ---
VOL_BREAKOUT_ATR_MULT=1.5
VOL_BREAKOUT_LOOKBACK=20
VOL_BREAKOUT_VOL_MULT=1.2
VOL_BREAKOUT_POSITION_SIZE=0.06
VOL_BREAKOUT_BB_LAG=3
```

### 7.6 Risk Layer v2

```dotenv
GLOBAL_KILL_SWITCH_ENABLED=true
GLOBAL_MAX_DRAWDOWN_FRAC=0.10      # 10% → para todo el bot

MAX_CRITICAL_ERRORS=5

MAX_SYMBOL_DRAWDOWN_FRAC=0.06
MAX_CAPITAL_COMMITTED_FRAC=0.15
DAILY_LOSS_LIMIT_FRAC=0.03
SYMBOL_COOLDOWN_AFTER_DAILY_LOSS_SECONDS=86400

RECOVERED_ATR_INTERVAL=1h
RECOVERED_ATR_PERIOD=14
RECOVERED_SL_ATR_MULT=2.0
RECOVERED_TP_ATR_MULT=2.0
RECOVERED_TRAILING_SL_ATR_MULT=2.0

CANCEL_OPEN_ORDERS_ON_STOP=1
```

### 7.7 Risk Layer v3 (opcional)

```dotenv
RISK_V3_POSITION_SIZER_ENABLED=false
RISK_V3_CORRELATION_ENABLED=false
RISK_V3_VAR_ENABLED=false
RISK_V3_SLIPPAGE_ENABLED=false
RISK_V3_EQUITY_REGIME_ENABLED=false
```

Ver `.env.example` para todos los parámetros de cada componente.

### 7.8 Market Data Process

```dotenv
MARKET_DATA_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,AVAXUSDT,BNBUSDT
MARKET_DATA_INTERVALS=1m,1h,2h,4h,1d,1w
MARKET_DATA_SYNC_EVERY_SECONDS=30

KLINES_BACKFILL_LIMIT=1000
KLINES_MAX_PAGES=200
KLINES_OVERLAP_CANDLES=2
```

### 7.9 Infraestructura

```dotenv
HEALTH_HOST=0.0.0.0
HEALTH_PORT=8001             # Flask health endpoint (evitar 8000 si hay conflicto)

EQUITY_SNAPSHOT_INTERVAL=300

TELEGRAM_BOT_TOKEN=          # opcional pero recomendado en producción
TELEGRAM_CHAT_ID=

LOG_LEVEL=INFO               # DEBUG para diagnóstico detallado
```

---

## 8. Instalación y ejecución

### Requisitos

- Python 3.12+
- PostgreSQL 14+
- Cuenta de Binance con API activada en TESTNET

### Paso a paso (entorno local)

```bash
# 1. Clonar y configurar entorno
git clone <repo-url>
cd bot-crypto-trading-BNB
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales (ver sección 7)

# 3. Crear el schema de PostgreSQL (idempotente)
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f db/schema.sql

# 4. [Terminal 1] Arrancar el proceso de datos (PRIMERO)
python market_data_process.py
# Esperar al menos 260 velas sincronizadas antes de arrancar el bot

# 5. [Terminal 2] Arrancar el bot de trading
python bot_trading_v5.py

# 6. Verificar estado
curl -s http://localhost:8001/health | python3 -m json.tool
```

### Verificar datos suficientes antes de arrancar el bot

```sql
-- Necesitas al menos 260 velas para que EMA200 esté disponible
SELECT symbol, interval, COUNT(*) AS n_candles,
       MIN(open_time) AS desde, MAX(open_time) AS hasta
FROM trading.market_klines
GROUP BY 1, 2
ORDER BY 1, 2;
```

### Log de arranque esperado

```
INFO  Bot Trading v2 iniciado. LOG_LEVEL=INFO
INFO  Clientes Binance inicializados: data=MAINNET trade=TESTNET
INFO  PostgreSQL listo: storage habilitado
INFO  Risk v3 init: sizer=False corr=False var=False ...
INFO  StrategyEngine inicializado: 5 estrategias — [ml_momentum, trend_following, ...]
INFO  PortfolioManager: buy_thr=1.00 veto_on_hold=False modo=HÍBRIDO ml_min_conf=0.40 scales=(0.5x/1.0x/1.5x)
INFO  ML model cargado: path=artifacts/model_momentum_v4_phase2.joblib threshold=0.600 mode=active
INFO  [BTCUSDT] Precio: XXXXX | EMA200: XXXXX | Regime: BEAR/BULL/LATERAL
```

---

## 9. Despliegue con Docker

El repositorio incluye `Dockerfile.bot`, `Dockerfile.market_data` y `docker-compose.yml`
para desplegar los dos procesos como contenedores independientes.

### Arquitectura de contenedores

```
┌─────────────────────────────────┐    ┌──────────────────────────────────┐
│  Contenedor: market-data        │    │  Contenedor: bot-trading          │
│  Dockerfile.market_data         │    │  Dockerfile.bot                   │
│                                 │    │                                   │
│  market_data_process.py         │    │  bot_trading_v5.py                │
│  db/ services/ repositories/    │    │  strategies/ artifacts/           │
│                                 │    │  risk_layer_v2.py                 │
│  health: pgrep market_data      │    │  health: GET /health :8001        │
└──────────────┬──────────────────┘    └──────────────┬────────────────────┘
               │                                      │
               └──────────── PostgreSQL ──────────────┘
                            (host externo)
```

### Comandos básicos

```bash
# Construir imágenes
docker compose build

# Arrancar ambos servicios en background
docker compose up -d

# Solo el proceso de datos (arrancar primero para poblar la DB)
docker compose up -d market-data

# Bot de trading (una vez que market-data esté healthy)
docker compose up -d bot-trading

# Ver logs en tiempo real
docker compose logs -f bot-trading
docker compose logs -f market-data

# Estado de los contenedores
docker compose ps

# Detener todo
docker compose down
```

### Notas de despliegue

- Ambos contenedores usan `network_mode: host` para acceder directamente a
  PostgreSQL en la IP del host (ej. `192.168.1.50`) sin configuración adicional.
- El directorio `logs/` y `artifacts/` se montan como volúmenes desde el host,
  lo que permite actualizar el modelo ML sin reconstruir la imagen.
- El `.env` **no se incluye en la imagen** (está en `.dockerignore`); debe
  existir en el host y es leído vía `env_file` en `docker-compose.yml`.
- `bot-trading` tiene `depends_on: market-data: condition: service_healthy`,
  por lo que esperará a que el proceso de datos esté operativo antes de arrancar.

---

## 10. Backtesting

El framework simula el bot vela por vela sobre datos históricos,
usando exactamente la misma lógica que en live (sin lookahead).

### Backtest multi-estrategia 2022–2025

Script dedicado con features cross-asset pre-computadas desde DB:

```bash
# Backtest 2022 con modo híbrido
python run_backtest_2022_multi.py \
    --start 2022-01-01 --end 2023-01-01 \
    --initial-cash 10000 \
    --hybrid-mode true \
    --ml-min-confidence 0.40 \
    --buy-threshold 1.0 \
    --output-dir backtest_out_2022_hybrid

# Opciones disponibles
python run_backtest_2022_multi.py --help
```

Parámetros clave:

| Flag | Default | Descripción |
|---|---|---|
| `--start` / `--end` | 2022-01-01 / 2023-01-01 | Rango de fechas |
| `--initial-cash` | 10000 | Capital inicial en USDT |
| `--hybrid-mode` | false | Activar ML como escalador |
| `--ml-min-confidence` | 0.40 | Gate mínimo de confianza ML |
| `--buy-threshold` | 1.0 | Score mínimo para entrar |
| `--veto-on-hold` | true | HOLD veta la entrada |
| `--output-dir` | backtest_out_2022_multi | Directorio de salida |

### Backtest con estrategia ML (BotV5StrategyAdapter)

```bash
python -m backtesting.run_backtest \
  --symbols BTCUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --strategy v5 \
  --output-dir backtest_out/

python -m backtesting.run_backtest --help
```

### Outputs

| Archivo | Contenido |
|---|---|
| `<output-dir>/equity_curve.csv` | Equity total, drawdown y posiciones por vela |
| `<output-dir>/trades.csv` | Detalle de cada trade: entry/exit, PnL, duración, motivo |

---

## 11. Tests

```bash
source .venv/bin/activate

# Todos los tests
pytest tests/ -v

# Tests rápidos (sin DB ni red)
pytest tests/test_strategy_engine_and_portfolio.py \
       tests/test_risk_layer_v2.py \
       tests/test_risk_layer_v3.py -v

# Tests de integración del bot v5
pytest tests/test_bot_trading_v5_integration.py -v
```

### Cobertura de tests

| Archivo de test | Qué cubre | Tests |
|---|---|---|
| `test_strategy_engine_and_portfolio.py` | StrategyEngine, PortfolioManager (modo híbrido + estándar), Signal, MarketState | 27 |
| `test_risk_layer_v2.py` | GlobalRisk, HealthMonitor, reconcile | — |
| `test_risk_layer_v3.py` | VaR, correlación, slippage, sizing, regime | — |
| `test_bot_trading_v5_integration.py` | `_build_v5_strategy_adapter`, `run_strategy` end-to-end | — |
| `test_estrategia_v5.py` | BotV5StrategyAdapter, prepare_indicators, generate_entry | — |

---

## 12. Monitoreo y observabilidad

### Health endpoint

```bash
curl http://localhost:8001/health
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
y también a stdout para integración con Docker (`docker compose logs -f bot-trading`).

```bash
# Seguir logs en tiempo real
tail -f logs/bot_trading.log

# Filtrar señales de entrada
grep "Señal entrada\|PortfolioManager" logs/bot_trading.log | tail -50

# Ver régimen y precio por ciclo
grep "Precio:" logs/bot_trading.log | tail -20
```

### Telegram

Si configuras `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`, el bot envía notificaciones de:
- COMPRA ejecutada (precio, tamaño, TP/SL, estrategias que dispararon)
- VENTA ejecutada (motivo: SL / TP / trailing / BEAR)
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

-- Últimos 10 trades con PnL
SELECT symbol, reason, buy_price, sell_price,
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
       COUNT(*) AS trades,
       SUM(realized_pnl) AS pnl_total,
       AVG(realized_pnl_pct) * 100 AS avg_pnl_pct,
       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)::float / COUNT(*) * 100 AS win_rate_pct
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

## 13. Estructura del repositorio

```
bot-crypto-trading-BNB/
│
├── bot_trading_v5.py              # Entrypoint del bot (orquestación principal)
├── market_data_process.py         # Proceso de ingesta OHLCV → PostgreSQL
│
├── strategies/                    # Sistema multi-estrategia v2
│   ├── base_strategy.py           # BaseStrategy, Signal, MarketState
│   ├── ml_strategy.py             # MLStrategy — expone raw_prob en HOLD para modo híbrido
│   ├── trend_strategy.py          # TrendStrategy (BTC/ETH, EMA50>EMA200+ADX)
│   ├── mean_reversion_strategy.py # MeanReversionStrategy (altcoins, BB+RSI)
│   ├── funding_arbitrage_strategy.py  # FundingArbitrageStrategy (filtro macro)
│   └── volatility_breakout_strategy.py
│
├── strategy_engine.py             # StrategyEngine.collect() → list[Signal]
├── portfolio_manager.py           # PortfolioManager — modo estándar + híbrido ML
├── strategies_multi.py            # MultiStrategyEngine legacy (ANY/MAJORITY/ALL)
│
├── estrategia_v5.py               # BotV5StrategyAdapter (ML + indicadores técnicos)
├── estrategia_multi.py            # MultiStrategyBacktestAdapter — fix df slice iloc[-2]
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
│   ├── market_klines_repo.py      # Lectura/escritura de klines
│   └── equity_repo.py             # Snapshots de equity
│
├── services/
│   └── market_klines_service.py   # Lectura de klines desde PostgreSQL
│
├── backtesting/
│   ├── bt_types.py                # Strategy, StrategyContext, EntrySignal, Position, Bar
│   ├── engine.py                  # BacktestEngine (simulación vela-por-vela)
│   ├── metrics.py                 # Sharpe, Sortino, max DD, profit factor
│   └── run_backtest.py            # CLI: --strategy v5/v4/v3_1/v2_2
│
├── run_backtest_2022_multi.py     # Backtest histórico multi-estrategia 2022-2025
│                                  # con cross-asset features desde DB y modo híbrido
│
├── artifacts/
│   └── model_momentum_v4_phase2.joblib  # Modelo ML activo (18 MB, gradient boosting)
│
├── tests/                         # Suite de tests unitarios e integración
│
├── tools/
│   ├── env_audit.py               # Auditoría de variables de entorno
│   └── kline_gap_audit.py         # Detección de gaps en market_klines
│
├── Dockerfile.bot                 # Imagen Docker para bot_trading_v5.py
├── Dockerfile.market_data         # Imagen Docker para market_data_process.py
├── docker-compose.yml             # Orquestación: market-data + bot-trading
├── .dockerignore                  # Excluye .env, .venv, backtest_out*, tests/
│
├── .env.example                   # Plantilla completa de variables de entorno
├── requirements.txt               # Dependencias Python (Python 3.12)
└── logs/
    └── bot_trading.log            # Rotativo 10 MB × 5 archivos
```
