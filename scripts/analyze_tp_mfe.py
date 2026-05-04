"""
Resume take_profit_pct configurado vs MFE real al cierre (trade_features + trades).

Ayuda a decidir si subir `take_profit_target` o el trailing: si actual_mfe >> take_profit_pct
en salidas por take_profit, hay margen.

Uso:
  python scripts/analyze_tp_mfe.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import DatabaseManager


async def main() -> None:
    db = DatabaseManager()
    await db.connect()
    try:
        rows = await db.get_closed_trade_tp_mfe_rows()
        if not rows:
            print("Sin filas: trades cerrados con take_profit_pct y actual_mfe en trade_features.")
            return
        tp_only = [r for r in rows if (r.get("exit_reason") or "") == "take_profit"]
        mfe_all = [float(r["actual_mfe"]) for r in rows]
        mfe_tp = [float(r["actual_mfe"]) for r in tp_only]
        tp_pct_tp = [float(r["take_profit_pct"]) for r in tp_only]
        print(f"Muestras (cerrados con features): {len(rows)}")
        print(f"  Salidas take_profit: {len(tp_only)}")
        if mfe_all:
            print(f"  actual_mfe (todos): mean={statistics.mean(mfe_all):.3f} median={statistics.median(mfe_all):.3f}")
        if mfe_tp and tp_pct_tp:
            print(
                f"  En take_profit: mean(actual_mfe)={statistics.mean(mfe_tp):.3f} "
                f"mean(take_profit_pct)={statistics.mean(tp_pct_tp):.3f}"
            )
            headroom = [float(r["actual_mfe"]) - float(r["take_profit_pct"]) for r in tp_only]
            print(
                f"  Headroom MFE - TP% en TP exits: mean={statistics.mean(headroom):.3f} "
                f"median={statistics.median(headroom):.3f}"
            )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
