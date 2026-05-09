"""
Backtest offline de la regla B (pullback mínimo respecto al máximo local reciente).

Usa precios guardados en parquet (`TimeseriesManager`, mismo origen que el monitor).
Para cada trade cerrado, en el instante de `entry_time` mira el máximo de `price_usd`
en la ventana [entry_time - lookback, entry_time] y calcula:

  drawdown_from_high = (max_local - entry_price) / max_local

donde max_local = max(max_en_ventana, entry_price) para no subestimar el techo.

La regla B **filtraría** (no habría entrado) si drawdown_from_high < min_drawdown_pct.

Uso:
  python scripts/backtest_entry_rule_b.py --days 30 --lookback 45 --min-dd 0.05
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import DatabaseManager, TimeseriesManager


def _parse_entry_dt(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    if len(s) >= 19:
        s = s[:19]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    pnls = [float(r.get("pnl_usd") or 0.0) for r in rows]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    return {
        "n": float(len(rows)),
        "sum_pnl": float(sum(pnls)),
        "win_rate": (len(wins) / len(rows)) if rows else 0.0,
        "avg_pnl": statistics.mean(pnls) if pnls else 0.0,
        "avg_win": statistics.mean(wins) if wins else 0.0,
        "avg_loss": statistics.mean(losses) if losses else 0.0,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest regla B: pullback vs máximo local")
    ap.add_argument("--days", type=int, default=30, help="Ventana de trades cerrados (entry_time)")
    ap.add_argument("--lookback", type=int, default=45, help="Minutos hacia atrás desde entry_time")
    ap.add_argument(
        "--min-dd",
        type=float,
        default=0.05,
        help="Drawdown mínimo exigido vs máximo local (ej. 0.05 = 5%%)",
    )
    args = ap.parse_args()

    db = DatabaseManager()
    ts = TimeseriesManager()
    await db.connect()
    try:
        cur = await db._connection.execute(
            """
            SELECT
              t.id AS trade_id,
              t.token_address,
              t.entry_time,
              t.entry_price,
              t.pnl_usd,
              t.exit_reason,
              t.label
            FROM trades t
            WHERE t.exit_time IS NOT NULL
              AND t.entry_time IS NOT NULL
              AND t.entry_price IS NOT NULL
              AND t.entry_time > datetime('now', ?)
            ORDER BY datetime(t.entry_time) ASC, t.id ASC
            """,
            (f"-{int(args.days)} days",),
        )
        rows = [dict(r) for r in await cur.fetchall()]
        if not rows:
            print("Sin trades cerrados con entry en el periodo.")
            return

        lookback = timedelta(minutes=int(args.lookback))
        min_dd = float(args.min_dd)

        kept: List[Dict[str, Any]] = []
        filtered: List[Dict[str, Any]] = []
        no_ts: List[Dict[str, Any]] = []

        for r in rows:
            entry_dt = _parse_entry_dt(r.get("entry_time"))
            if entry_dt is None:
                no_ts.append(r)
                kept.append(r)
                continue
            start = entry_dt - lookback
            df = await ts.get_token_prices(
                str(r["token_address"]),
                start_date=start,
                end_date=entry_dt,
            )
            if df is None or df.empty:
                no_ts.append(r)
                kept.append(r)
                continue
            dfc = df.copy()
            dfc["timestamp"] = pd.to_datetime(dfc["timestamp"], utc=False)
            et = pd.Timestamp(entry_dt)
            st = pd.Timestamp(start)
            dfc = dfc[(dfc["timestamp"] >= st) & (dfc["timestamp"] <= et)]
            if dfc.empty:
                no_ts.append(r)
                kept.append(r)
                continue
            mx = float(dfc["price_usd"].max())
            entry_px = float(r.get("entry_price") or 0.0)
            if entry_px <= 0 or mx <= 0:
                no_ts.append(r)
                kept.append(r)
                continue
            hi = max(mx, entry_px)
            dd = (hi - entry_px) / hi if hi > 0 else 0.0
            r2 = {**r, "dd_vs_local_high": dd, "local_high": hi}
            if dd < min_dd:
                filtered.append(r2)
            else:
                kept.append(r2)

        base = _summarize(rows)
        after = _summarize(kept)
        cut = _summarize(filtered)

        print("\n[backtest_entry_rule_b]")
        print(f"  days={args.days} lookback_min={args.lookback} min_drawdown_from_high={min_dd:.3f}\n")
        print(f"  trades sin serie util en ventana (no filtra): {len(no_ts)}")
        print("  BASE (todos los trades del periodo):")
        print(
            f"    n={int(base['n'])} sum_pnl={base['sum_pnl']:.2f} "
            f"win_rate={base['win_rate']:.1%} avg_pnl={base['avg_pnl']:.2f}"
        )
        print("  SERIAN FILTRADOS (compra demasiado cerca del maximo local):")
        print(
            f"    n={int(cut['n'])} sum_pnl={cut['sum_pnl']:.2f} "
            f"win_rate={cut['win_rate']:.1%} avg_pnl={cut['avg_pnl']:.2f}"
        )
        print("  RESTO (tras aplicar B; sin datos = se mantiene el trade):")
        print(
            f"    n={int(after['n'])} sum_pnl={after['sum_pnl']:.2f} "
            f"win_rate={after['win_rate']:.1%} avg_pnl={after['avg_pnl']:.2f}"
        )
        delta = after["sum_pnl"] - base["sum_pnl"]
        print(f"\n  delta_sum_pnl (after - base): {delta:.2f}")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
