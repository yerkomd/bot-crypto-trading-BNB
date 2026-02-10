# PostgreSQL storage (v3.1)

Este bot persiste **todo el estado** en PostgreSQL (schema `trading`). No se usan CSV para posiciones ni trades.

## Arquitectura desacoplada (obligatoria)

1) **Market Data Process (NO trading)**
- Descarga velas OHLCV desde **Binance MAINNET**.
- Hace backfill + sync incremental.
- Inserta en `trading.market_klines`.
- Entry-point: `market_data_process.py`.

2) **Trading Bot (tiempo real)**
- Lee velas históricas **solo desde PostgreSQL** para indicadores/señales.
- Ejecuta órdenes y monitorea SL/TP/trailing con **Binance TESTNET** (precio live).

Regla crítica: PostgreSQL es fuente única para histórico, **no** para precio live.

## 1) Variables de entorno (OBLIGATORIAS)

En `.env` (ver `.env.example`):

```dotenv
USE_DATABASE=true
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=bot_trading
POSTGRES_USER=trader
POSTGRES_PASSWORD=changeme

# Snapshots cada N segundos
EQUITY_SNAPSHOT_INTERVAL=300
```

## 2) Esquema / Tablas

- SQL idempotente: `db/schema.sql`
- Se aplica automáticamente al iniciar el bot (`db/schema.py`).

Si ves un error tipo `permission denied for database bot_trading`, tu usuario de DB no tiene permisos para crear el schema/tablas. Solución (una sola vez, con usuario admin/owner): aplica `db/bootstrap_admin.sql` y luego corre el healthcheck con `--ensure-schema`.

Tablas:
- `trading.open_positions`
- `trading.trade_history`
- `trading.equity_snapshots`
- `trading.market_klines` (OHLCV histórico, incremental)

## 3) Market Data Process (recomendado)

Desde la carpeta del bot:

```bash
python market_data_process.py
```

Variables recomendadas:
- `MARKET_DATA_SYMBOLS=BTCUSDT,ETHUSDT`
- `MARKET_DATA_INTERVALS=1h,4h`
- `MARKET_DATA_SYNC_EVERY_SECONDS=30`

## 3) Ejemplos (SELECT / INSERT)

```sql
-- Ver posiciones abiertas
SELECT * FROM trading.open_positions WHERE symbol='BNBUSDT' ORDER BY opened_at;

-- Ver trades cerrados
SELECT symbol, side, reason, realized_pnl, executed_at
FROM trading.trade_history
WHERE symbol='BNBUSDT'
ORDER BY executed_at DESC
LIMIT 100;

-- Equity curve
SELECT timestamp, equity_total
FROM trading.equity_snapshots
ORDER BY timestamp;

-- Últimas velas (para indicadores / ML)
SELECT open_time, open, high, low, close, volume
FROM trading.market_klines
WHERE symbol='BNBUSDT' AND interval='1h'
ORDER BY open_time DESC
LIMIT 300;
```

Ejemplo Python (mismo patrón que usa el bot):

```python
from db.connection import init_db_from_env
from db.schema import ensure_schema
from repositories.open_positions_repo import OpenPositionsRepository

db = init_db_from_env()
ensure_schema(db)
repo = OpenPositionsRepository(db)

# SELECT
pos = repo.list_by_symbol("BNBUSDT")
print(pos)

# INSERT
new_id = repo.insert({
	"symbol": "BNBUSDT",
	"buy_price": 300.0,
	"amount": 0.01,
	"take_profit": 315.0,
	"stop_loss": 290.0,
	"regime": "BULL",
	"opened_at": "2026-02-04T00:00:00Z",
})
print("inserted id", new_id)
```

## 4) Operación

- El bot inicializa DB al arrancar y reintenta conexión si falla.
- Si hay un fallo temporal de DB, el bot no cae: registra error y reintenta.

## 5) Healthcheck (recomendado antes de dinero real)

Desde la carpeta del bot:

```bash
/home/melgary/Proyectos/.venv/bin/python tools/db_healthcheck.py --ensure-schema
```

Para automatización/monitoring (salida JSON):

```bash
/home/melgary/Proyectos/.venv/bin/python tools/db_healthcheck.py --json
```

Notas:
- El check de escritura usa una **tabla TEMP** y hace `ROLLBACK` (no ensucia datos reales).
- Código de salida: `0` ok, `2` error de configuración/conexión, `4` esquema incompleto.
