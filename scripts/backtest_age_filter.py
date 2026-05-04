"""
Contrasta PnL agregado de trades cerrados si hubieran existido filtros de edad distintos.

Usa `age_hours` guardado en `trade_features` al abrir el trade (no re-simula precios).

Uso:
  python scripts/backtest_age_filter.py
  python scripts/backtest_age_filter.py --min-age 5 15 25
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import DatabaseManager


def _allowed(age_hours: float, min_age_minutes: float) -> bool:
    return age_hours * 60.0 >= min_age_minutes


def _summarize(rows: List[dict], thresholds: Sequence[float]) -> None:
    print("\n--- Contraste por umbral min_token_age (minutos) ---\n")
    for m in thresholds:
        pnl = 0.0
        n = 0
        skipped = 0
        for r in rows:
            ah = float(r["age_hours"])
            if _allowed(ah, m):
                pnl += float(r.get("pnl_usd") or 0.0)
                n += 1
            else:
                skipped += 1
        print(f"  min_age >= {m:>4.0f} min  |  trades contados: {n:>4}  skipped: {skipped:>4}  sum_pnl_usd: {pnl:>12.2f}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest offline de filtro de edad sobre trades históricos")
    parser.add_argument(
        "--min-age",
        type=float,
        nargs="*",
        default=[5.0, 15.0, 25.0],
        help="Minutos mínimos de edad al entry a simular (default: 5 15 25)",
    )
    args = parser.parse_args()

    db = DatabaseManager()
    await db.connect()
    try:
        rows = await db.get_closed_trades_with_entry_age_hours()
        if not rows:
            print("No hay trades cerrados con age_hours en trade_features.")
            return
        print(f"Trades con age_hours al entry: {len(rows)}")
        _summarize(rows, args.min_age)
        print(
            "\nInterpretación: 'skipped' son trades que el bot habría bloqueado con ese min_age; "
            "sum_pnl_usd es la suma de pnl_usd solo de los que pasarían el filtro."
        )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
