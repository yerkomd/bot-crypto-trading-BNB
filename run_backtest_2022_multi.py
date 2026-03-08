"""
run_backtest_2022_multi.py — Backtest 2022 con MultiStrategyBacktestAdapter (multi-estrategia v2).

Carga BTCUSDT e ETHUSDT 1h desde PostgreSQL para el año 2022, pre-computa los
features cross-asset (eth_correlation_30, btc_dominance, market_breadth) e inyecta
esos valores en el df antes de llamar al motor de backtesting, resolviendo el problema
de que CoinGecko no tiene datos históricos para btc_dominance en tiempo de backtest.

Uso:
    python run_backtest_2022_multi.py [--start 2022-01-01] [--end 2023-01-01] \
        [--initial-cash 10000] [--fee-rate 0.001] [--output-dir backtest_out_2022_multi]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── Setup path ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / ".env", override=False)

from backtesting import BacktestEngine, BacktestConfig
from backtesting.bt_types import Strategy, StrategyContext, EntrySignal, Position, Bar
from db.connection import init_db_from_env
from services.market_klines_service import read_klines_df_range
from estrategia_multi import MultiStrategyBacktestAdapter


# ── Cross-asset feature injection ────────────────────────────────────────────

def _compute_cross_asset_features(
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
    btc_dominance_constant: float = 43.0,
    corr_window: int = 30,
) -> pd.DataFrame:
    """Agrega columnas cross-asset a df_btc, alineadas por timestamp.

    - eth_correlation_30: correlación móvil 30 velas entre retornos BTC y ETH
    - btc_dominance: constante histórica 2022 (promedio ~43%)
    - market_breadth: fracción de las dos monedas con retorno positivo (proxy simple)

    Returns df_btc enriquecido (copia).
    """
    df = df_btc.copy()

    # ── Retornos BTC ──────────────────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_btc_ret"] = df["close"].pct_change()

    # ── Alinear retornos ETH ───────────────────────────────────────────────────
    eth = df_eth.sort_values("timestamp").copy()
    eth["_eth_ret"] = eth["close"].pct_change()
    eth = eth[["timestamp", "_eth_ret", "close"]].rename(
        columns={"close": "_eth_close"}
    )

    df = df.merge(eth, on="timestamp", how="left")
    df["_eth_ret"] = df["_eth_ret"].ffill()

    # ── eth_correlation_30 ────────────────────────────────────────────────────
    df["eth_correlation_30"] = (
        df["_btc_ret"]
        .rolling(corr_window, min_periods=max(5, corr_window // 3))
        .corr(df["_eth_ret"])
    )
    # Rellenar NaN iniciales con 0 (sin correlación disponible)
    df["eth_correlation_30"] = df["eth_correlation_30"].fillna(0.0)

    # ── btc_dominance: constante ───────────────────────────────────────────────
    df["btc_dominance"] = btc_dominance_constant

    # ── market_breadth: proxy con BTC+ETH ─────────────────────────────────────
    df["market_breadth"] = (
        (df["_btc_ret"].fillna(0) > 0).astype(float)
        + (df["_eth_ret"].fillna(0) > 0).astype(float)
    ) / 2.0

    # Limpiar columnas temporales
    df = df.drop(columns=["_btc_ret", "_eth_ret", "_eth_close"], errors="ignore")

    return df


# ── Custom adapter que inyecta cross-asset features ─────────────────────────

class MultiStrategy2022Adapter:
    """Wrapper alrededor de MultiStrategyBacktestAdapter que inyecta features cross-asset
    en el df antes de prepare_indicators, para permitir un backtest histórico sin
    acceso en vivo a CoinGecko.
    """

    def __init__(
        self,
        cross_asset_df: pd.DataFrame,
        position_size_frac: float = 0.03,
        buy_threshold: float = 1.0,
        sell_threshold: float = 1.0,
        veto_on_hold: bool = True,
        ml_hybrid_mode: bool = False,
        ml_min_confidence: float = 0.40,
        ml_size_scale_low: float = 0.5,
        ml_size_scale_mid: float = 1.0,
        ml_size_scale_high: float = 1.5,
    ):
        from strategy_engine import build_default_engine
        from portfolio_manager import PortfolioManager

        self._inner = MultiStrategyBacktestAdapter(position_size_frac=position_size_frac)
        engine = build_default_engine()
        self._inner._engine = engine
        self._inner._portfolio = PortfolioManager(
            strategies=engine.strategies,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            veto_on_hold=veto_on_hold,
            ml_hybrid_mode=ml_hybrid_mode,
            ml_min_confidence=ml_min_confidence,
            ml_size_scale_low=ml_size_scale_low,
            ml_size_scale_mid=ml_size_scale_mid,
            ml_size_scale_high=ml_size_scale_high,
        )
        # Indexed by timestamp for fast O(1) lookup during backtest iteration
        self._cross_asset = cross_asset_df.set_index("timestamp")
        self._cross_columns = ["eth_correlation_30", "btc_dominance", "market_breadth"]

    def _inject_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inyecta columnas cross-asset en df alineando por timestamp."""
        df = df.copy()
        for col in self._cross_columns:
            if col not in df.columns and col in self._cross_asset.columns:
                # Map via timestamp index
                ts_idx = df["timestamp"] if "timestamp" in df.columns else df.index
                df[col] = ts_idx.map(
                    self._cross_asset[col].to_dict()
                ).fillna(method="ffill" if False else None)
                df[col] = df[col].fillna(
                    self._cross_asset[col].mean()
                    if col != "btc_dominance"
                    else 43.0
                )
        return df

    def prepare_indicators(self, *, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        enriched = self._inject_features(df)
        return self._inner.prepare_indicators(symbol=symbol, df=enriched)

    def generate_entry(self, ctx: StrategyContext) -> EntrySignal:
        return self._inner.generate_entry(ctx)

    def compute_risk_levels(
        self,
        *,
        symbol: str,
        regime: str,
        buy_price: float,
        indicators_row: dict,
    ) -> tuple:
        return self._inner.compute_risk_levels(
            symbol=symbol,
            regime=regime,
            buy_price=buy_price,
            indicators_row=indicators_row,
        )

    def update_trailing(
        self,
        *,
        symbol: str,
        position: Position,
        bar: Bar,
        indicators_row: dict,
    ) -> None:
        return self._inner.update_trailing(
            symbol=symbol,
            position=position,
            bar=bar,
            indicators_row=indicators_row,
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Backtest 2022 con MultiStrategyBacktestAdapter (multi-estrategia v2)"
    )
    p.add_argument("--start", type=str, default="2022-01-01T00:00:00")
    p.add_argument("--end", type=str, default="2023-01-01T00:00:00")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--eth-symbol", type=str, default="ETHUSDT")
    p.add_argument("--interval", type=str, default="1h")
    p.add_argument("--initial-cash", type=float, default=10000.0)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--position-size", type=float, default=0.03)
    p.add_argument("--btc-dominance", type=float, default=43.0,
                   help="Constante de dominancia BTC para 2022 (promedio histórico ~43%)")
    p.add_argument("--buy-threshold", type=float, default=1.0,
                   help="Score mínimo para aprobar BUY (PORTFOLIO_BUY_THRESHOLD)")
    p.add_argument("--sell-threshold", type=float, default=1.0,
                   help="Score mínimo para aprobar SELL (PORTFOLIO_SELL_THRESHOLD)")
    p.add_argument("--veto-on-hold", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True,
                   help="Si cualquier HOLD veta la entrada (true/false, default true)")
    # Modo híbrido
    p.add_argument("--hybrid-mode", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=False,
                   help="Activar modo híbrido: ML escala tamaño en vez de vetar (default false)")
    p.add_argument("--ml-min-confidence", type=float, default=0.40,
                   help="Gate mínimo de confianza ML en modo híbrido (default 0.40)")
    p.add_argument("--ml-scale-low", type=float, default=0.5,
                   help="Factor de escala para conf ∈ [min, 0.55) (default 0.5)")
    p.add_argument("--ml-scale-mid", type=float, default=1.0,
                   help="Factor de escala para conf ∈ [0.55, 0.70) (default 1.0)")
    p.add_argument("--ml-scale-high", type=float, default=1.5,
                   help="Factor de escala para conf ≥ 0.70 (default 1.5)")
    p.add_argument("--output-dir", type=str, default="backtest_out_2022_multi")
    args = p.parse_args(argv)

    start_dt = pd.to_datetime(args.start, utc=True).to_pydatetime().replace(tzinfo=None)
    end_dt = pd.to_datetime(args.end, utc=True).to_pydatetime().replace(tzinfo=None)

    print(f"[CONFIG] Symbol: {args.symbol}, interval: {args.interval}")
    print(f"[CONFIG] Período: {start_dt} -> {end_dt}")
    print(f"[CONFIG] Cash inicial: ${args.initial_cash:,.0f}, fee: {args.fee_rate*100:.2f}%")
    print(f"[CONFIG] BTC dominance constante: {args.btc_dominance}%")

    # ── Cargar datos de PostgreSQL ────────────────────────────────────────────
    print("\n[DB] Conectando a PostgreSQL...")
    db = init_db_from_env(require_use_database=False)

    print(f"[DB] Cargando {args.symbol} {args.interval} 2022...")
    df_btc = read_klines_df_range(
        db=db,
        symbol=args.symbol,
        interval=args.interval,
        start=start_dt,
        end=end_dt,
    )
    if df_btc is None or df_btc.empty:
        print(f"[ERROR] No hay datos para {args.symbol} en el período especificado.")
        return 1
    print(f"[DB] {args.symbol}: {len(df_btc)} velas ({df_btc['timestamp'].iloc[0]} -> {df_btc['timestamp'].iloc[-1]})")

    print(f"[DB] Cargando {args.eth_symbol} {args.interval} 2022 (cross-asset)...")
    df_eth = read_klines_df_range(
        db=db,
        symbol=args.eth_symbol,
        interval=args.interval,
        start=start_dt,
        end=end_dt,
    )
    if df_eth is None or df_eth.empty:
        print(f"[WARN] No hay datos ETH — eth_correlation_30 será 0.0 en todo el período.")
        df_eth = pd.DataFrame(columns=["timestamp", "close"])
        df_eth["timestamp"] = df_btc["timestamp"]
        df_eth["close"] = df_btc["close"]  # Fallback: correlación perfecta con sí mismo

    print(f"[DB] {args.eth_symbol}: {len(df_eth)} velas")

    # ── Pre-computar features cross-asset ─────────────────────────────────────
    print("\n[FEATURES] Computando eth_correlation_30, btc_dominance, market_breadth...")
    df_btc_enriched = _compute_cross_asset_features(
        df_btc=df_btc,
        df_eth=df_eth,
        btc_dominance_constant=args.btc_dominance,
    )
    cross_cols = ["eth_correlation_30", "btc_dominance", "market_breadth"]
    for col in cross_cols:
        sample = df_btc_enriched[col].describe()
        print(f"  {col}: mean={sample['mean']:.4f}, min={sample['min']:.4f}, max={sample['max']:.4f}")

    # ── Construir adaptador ───────────────────────────────────────────────────
    print("\n[STRATEGY] Construyendo MultiStrategy2022Adapter...")
    # Extract cross-asset columns for injection
    cross_asset_df = df_btc_enriched[["timestamp"] + cross_cols].copy()

    if args.hybrid_mode:
        print(f"[PORTFOLIO] MODO HÍBRIDO: ml_min_conf={args.ml_min_confidence} "
              f"scale=({args.ml_scale_low}x/{args.ml_scale_mid}x/{args.ml_scale_high}x) "
              f"buy_thr={args.buy_threshold}")
    else:
        print(f"[PORTFOLIO] Modo estándar: buy_threshold={args.buy_threshold} veto_on_hold={args.veto_on_hold}")
    strategy = MultiStrategy2022Adapter(
        cross_asset_df=cross_asset_df,
        position_size_frac=args.position_size,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        veto_on_hold=args.veto_on_hold,
        ml_hybrid_mode=args.hybrid_mode,
        ml_min_confidence=args.ml_min_confidence,
        ml_size_scale_low=args.ml_scale_low,
        ml_size_scale_mid=args.ml_scale_mid,
        ml_size_scale_high=args.ml_scale_high,
    )
    print("[STRATEGY] Adapter listo.")

    # ── Ejecutar backtest ─────────────────────────────────────────────────────
    config = BacktestConfig(
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        max_positions_per_symbol=1,
        min_notional=10.0,
    )

    engine = BacktestEngine(strategy=strategy, config=config)
    out_dir = Path(args.output_dir)

    print(f"\n[BACKTEST] Iniciando backtest ({len(df_btc_enriched)} velas)...")
    res = engine.run(
        data_by_symbol={args.symbol: df_btc_enriched},
        output_dir=out_dir,
    )

    # ── Resultados ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTADOS BACKTEST 2022 — Multi-estrategia v2")
    print("=" * 60)
    print(f"Equity curve: {res.equity_curve_path}")
    print(f"Trades:       {res.trades_path}")
    print("\nMétricas:")
    print(json.dumps(asdict(res.metrics), indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
