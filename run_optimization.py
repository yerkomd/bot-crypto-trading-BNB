"""
run_optimization.py — Optimización de parámetros de riesgo mediante backtesting iterativo.

Corre una grilla de combinaciones sobre 2022, 2024 y 2025, calcula un score
compuesto y muestra el ranking de las mejores configuraciones.

Score compuesto (mayor = mejor):
  score = return_pct × 0.35
        + profit_factor × 0.30
        - max_drawdown × 0.20
        + win_rate × 0.15

Uso:
  python run_optimization.py               # grilla completa (~40 combos × 3 años)
  python run_optimization.py --quick       # grilla reducida para validación rápida
  python run_optimization.py --top 5       # mostrar solo top 5
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Grilla de parámetros ──────────────────────────────────────────────────────

GRID_FULL = {
    "tp_mult":        [2.5, 3.0, 3.5, 4.0],
    "sl_mult":        [1.0, 1.2, 1.5],
    "trailing_mult":  [1.5, 2.0, 2.5],
    "ml_min_conf":    [0.35, 0.40, 0.45],
    "buy_threshold":  [1.0, 1.5],
}

# Grilla reducida: solo los ejes más impactantes (validación rápida)
GRID_QUICK = {
    "tp_mult":        [2.5, 3.0, 3.5],
    "sl_mult":        [1.0, 1.2, 1.5],
    "trailing_mult":  [1.5, 2.0],
    "ml_min_conf":    [0.40, 0.45],
    "buy_threshold":  [1.0],
}

YEARS = [
    ("2022", "2022-01-01", "2023-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
]

# Pesos del score compuesto
W_RETURN = 0.35
W_PF     = 0.30
W_DD     = 0.20   # se resta
W_WR     = 0.15


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class YearResult:
    year: str
    trades: int
    win_rate: float
    return_pct: float
    max_dd: float
    profit_factor: float
    sharpe: float
    expectancy: float


@dataclass
class ComboResult:
    tp_mult: float
    sl_mult: float
    trailing_mult: float
    ml_min_conf: float
    buy_threshold: float
    years: list[YearResult]
    score: float = 0.0

    @property
    def avg_return(self) -> float:
        return sum(y.return_pct for y in self.years) / len(self.years)

    @property
    def avg_dd(self) -> float:
        return sum(y.max_dd for y in self.years) / len(self.years)

    @property
    def avg_pf(self) -> float:
        vals = [y.profit_factor for y in self.years if y.profit_factor < 99]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_wr(self) -> float:
        vals = [y.win_rate for y in self.years if y.trades > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_trades(self) -> float:
        return sum(y.trades for y in self.years) / len(self.years)

    @property
    def ratio(self) -> float:
        return self.tp_mult / self.sl_mult


def compute_score(years: list[YearResult]) -> float:
    """Score compuesto normalizado entre los años evaluados."""
    if not years or all(y.trades == 0 for y in years):
        return -999.0
    active = [y for y in years if y.trades > 0]
    avg_ret = sum(y.return_pct for y in active) / len(active)
    avg_pf  = sum(min(y.profit_factor, 5.0) for y in active) / len(active)
    avg_dd  = sum(y.max_dd for y in active) / len(active)
    avg_wr  = sum(y.win_rate for y in active) / len(active)
    return (avg_ret * W_RETURN
            + avg_pf  * W_PF
            - avg_dd  * W_DD
            + avg_wr  * W_WR)


# ── Runner de un backtest ─────────────────────────────────────────────────────

def run_single(
    *,
    start: str,
    end: str,
    tp_mult: float,
    sl_mult: float,
    trailing_mult: float,
    ml_min_conf: float,
    buy_threshold: float,
    out_dir: str,
    initial_cash: float = 10_000.0,
) -> Optional[YearResult]:
    """Ejecuta un backtest y retorna YearResult o None si falla."""
    import subprocess, json

    env = os.environ.copy()
    env["BT_TP_ATR_MULT"]      = str(tp_mult)
    env["BT_SL_ATR_MULT"]      = str(sl_mult)
    env["BT_TRAILING_ATR_MULT"] = str(trailing_mult)

    cmd = [
        sys.executable, "run_backtest_2022_multi.py",
        "--start", start, "--end", end,
        "--initial-cash", str(initial_cash),
        "--hybrid-mode", "true",
        "--ml-min-confidence", str(ml_min_conf),
        "--buy-threshold", str(buy_threshold),
        "--veto-on-hold", "false",
        "--output-dir", out_dir,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env
        )
        if result.returncode != 0:
            return None

        trades_path = Path(out_dir) / "trades.csv"
        equity_path = Path(out_dir) / "equity_curve.csv"
        if not trades_path.exists() or not equity_path.exists():
            return None

        trades = pd.read_csv(trades_path)
        equity = pd.read_csv(equity_path)

        n = len(trades)
        if n == 0:
            return YearResult(year=start[:4], trades=0, win_rate=0.0,
                              return_pct=0.0, max_dd=0.0, profit_factor=0.0,
                              sharpe=0.0, expectancy=0.0)

        wr       = (trades["pnl"] > 0).mean() * 100
        ret      = (equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1) * 100
        dd       = equity["drawdown"].max() * 100
        wins_sum = trades[trades["pnl"] > 0]["pnl"].sum()
        loss_sum = -trades[trades["pnl"] <= 0]["pnl"].sum()
        pf       = wins_sum / loss_sum if loss_sum > 0 else 99.0
        exp      = trades["pnl"].mean()
        eq_ret   = equity["equity"].pct_change().dropna()
        sharpe   = (eq_ret.mean() / eq_ret.std() * (8760 ** 0.5)) if eq_ret.std() > 0 else 0.0

        return YearResult(year=start[:4], trades=n, win_rate=wr,
                          return_pct=ret, max_dd=dd, profit_factor=pf,
                          sharpe=sharpe, expectancy=exp)
    except Exception:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="Optimización de parámetros de riesgo")
    p.add_argument("--quick",       action="store_true", help="Grilla reducida")
    p.add_argument("--top",         type=int,   default=10,     help="Mostrar top N combos")
    p.add_argument("--initial-cash",type=float, default=10_000, help="Capital inicial")
    p.add_argument("--out-base",    type=str,   default="opt_out", help="Directorio base para outputs")
    args = p.parse_args(argv)

    grid = GRID_QUICK if args.quick else GRID_FULL
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    total = len(combos) * len(YEARS)
    print(f"\n{'='*70}")
    print(f"OPTIMIZACIÓN DE PARÁMETROS — {'GRILLA RÁPIDA' if args.quick else 'GRILLA COMPLETA'}")
    print(f"  Combinaciones: {len(combos)}  Años: {len(YEARS)}  Backtests: {total}")
    print(f"  Parámetros: {', '.join(keys)}")
    print(f"{'='*70}\n")

    results: list[ComboResult] = []
    run_n = 0

    for combo_vals in combos:
        params = dict(zip(keys, combo_vals))
        tp_mult       = params["tp_mult"]
        sl_mult       = params["sl_mult"]
        trailing_mult = params["trailing_mult"]
        ml_min_conf   = params["ml_min_conf"]
        buy_thr       = params["buy_threshold"]
        ratio         = tp_mult / sl_mult

        year_results: list[YearResult] = []

        for year, start, end in YEARS:
            run_n += 1
            tag = f"tp{tp_mult}_sl{sl_mult}_trail{trailing_mult}_conf{ml_min_conf}_bthr{buy_thr}"
            out_dir = f"{args.out_base}/{tag}/{year}"

            print(f"  [{run_n:3}/{total}] {year} tp={tp_mult} sl={sl_mult} "
                  f"trail={trailing_mult} conf={ml_min_conf} bthr={buy_thr} "
                  f"(ratio={ratio:.2f}x)... ", end="", flush=True)

            yr = run_single(
                start=start, end=end,
                tp_mult=tp_mult, sl_mult=sl_mult, trailing_mult=trailing_mult,
                ml_min_conf=ml_min_conf, buy_threshold=buy_thr,
                out_dir=out_dir, initial_cash=args.initial_cash,
            )

            if yr is None:
                print("ERROR")
                yr = YearResult(year=year, trades=0, win_rate=0, return_pct=0,
                                max_dd=0, profit_factor=0, sharpe=0, expectancy=0)
            else:
                print(f"trades={yr.trades:3}  ret={yr.return_pct:+6.2f}%  "
                      f"DD={yr.max_dd:5.2f}%  PF={yr.profit_factor:.2f}  "
                      f"WR={yr.win_rate:.1f}%")
            year_results.append(yr)

        score = compute_score(year_results)
        results.append(ComboResult(
            tp_mult=tp_mult, sl_mult=sl_mult, trailing_mult=trailing_mult,
            ml_min_conf=ml_min_conf, buy_threshold=buy_thr,
            years=year_results, score=score,
        ))

    # ── Ranking ───────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r.score, reverse=True)

    print(f"\n{'='*100}")
    print(f"RANKING TOP {args.top} — Score = Ret×{W_RETURN} + PF×{W_PF} - DD×{W_DD} + WR×{W_WR}")
    print(f"{'='*100}")
    header = (f"{'#':>3}  {'TP':>4} {'SL':>4} {'Trail':>5} {'Ratio':>5} "
              f"{'Conf':>5} {'Bthr':>4} │ "
              f"{'AvgRet%':>8} {'AvgDD%':>7} {'AvgPF':>6} {'AvgWR%':>7} "
              f"{'Trades':>6} │ {'Score':>7}")
    print(header)
    print("-" * 100)

    for i, r in enumerate(results[:args.top], 1):
        print(f"{i:>3}  {r.tp_mult:>4.1f} {r.sl_mult:>4.1f} {r.trailing_mult:>5.1f} "
              f"{r.ratio:>5.2f} {r.ml_min_conf:>5.2f} {r.buy_threshold:>4.1f} │ "
              f"{r.avg_return:>+8.2f}% {r.avg_dd:>6.2f}% {r.avg_pf:>6.2f} "
              f"{r.avg_wr:>6.1f}% {r.avg_trades:>6.0f} │ {r.score:>7.3f}")

    # Detalle del mejor combo por año
    best = results[0]
    print(f"\n{'='*100}")
    print(f"MEJOR CONFIGURACIÓN — Score: {best.score:.3f}")
    print(f"  BT_TP_ATR_MULT={best.tp_mult}  BT_SL_ATR_MULT={best.sl_mult}  "
          f"BT_TRAILING_ATR_MULT={best.trailing_mult}  "
          f"ML_MIN_CONFIDENCE={best.ml_min_conf}  PORTFOLIO_BUY_THRESHOLD={best.buy_threshold}")
    print(f"{'='*100}")
    print(f"{'Año':<6} {'Trades':>7} {'WinRate':>8} {'Return%':>9} {'MaxDD%':>7} "
          f"{'PF':>6} {'Sharpe':>7} {'Expectancy$':>12}")
    print("-" * 70)
    for y in best.years:
        print(f"{y.year:<6} {y.trades:>7} {y.win_rate:>7.1f}% {y.return_pct:>+8.2f}% "
              f"{y.max_dd:>6.2f}% {y.profit_factor:>6.2f} {y.sharpe:>+7.2f} "
              f"${y.expectancy:>10.2f}")

    # Exportar CSV completo
    rows = []
    for r in results:
        base = {
            "tp_mult": r.tp_mult, "sl_mult": r.sl_mult,
            "trailing_mult": r.trailing_mult, "ratio": r.ratio,
            "ml_min_conf": r.ml_min_conf, "buy_threshold": r.buy_threshold,
            "avg_return": r.avg_return, "avg_dd": r.avg_dd,
            "avg_pf": r.avg_pf, "avg_wr": r.avg_wr,
            "avg_trades": r.avg_trades, "score": r.score,
        }
        for y in r.years:
            base[f"ret_{y.year}"] = y.return_pct
            base[f"dd_{y.year}"] = y.max_dd
            base[f"pf_{y.year}"] = y.profit_factor
            base[f"wr_{y.year}"] = y.win_rate
        rows.append(base)

    csv_path = Path(args.out_base) / "optimization_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(csv_path, index=False)
    print(f"\nResultados completos exportados: {csv_path}")
    print(f"\n.env variables recomendadas para el mejor combo:")
    print(f"  BT_TP_ATR_MULT={best.tp_mult}")
    print(f"  BT_SL_ATR_MULT={best.sl_mult}")
    print(f"  BT_TRAILING_ATR_MULT={best.trailing_mult}")
    print(f"  ML_MIN_CONFIDENCE={best.ml_min_conf}")
    print(f"  PORTFOLIO_BUY_THRESHOLD={best.buy_threshold}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
