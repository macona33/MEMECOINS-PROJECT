"""
Auditoría de seguridad de mints vs historial de trades (DB local o copia del VPS).

Lista `token_program` / `has_freeze_authority` cacheados en `tokens` para cada mint
que aparece en `trades`. Útil para cruzar con logs "Safety(mint)" o ventas
"Account is frozen".

Uso:
  python scripts/audit_token_safety.py
  python scripts/audit_token_safety.py --mint CJMtHyHVMUCTCM8x8Jquz26BFENMv1LHRsMg2Y5ToGSR
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import DatabaseManager


async def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar token_program / freeze cache vs trades")
    parser.add_argument("--mint", type=str, default=None, help="Detalle de una dirección (mint o fila en tokens)")
    parser.add_argument("--limit", type=int, default=500, help="Máximo de mints distintos a listar")
    args = parser.parse_args()

    db = DatabaseManager()
    await db.connect()
    try:
        if args.mint:
            row = await db.get_token(args.mint)
            safe = await db.get_token_safety(args.mint)
            print(f"\nDirección: {args.mint}")
            if row:
                print(f"  tokens.symbol: {row.get('symbol')}")
                print(f"  tokens.token_program: {row.get('token_program')}")
                print(f"  tokens.has_freeze_authority: {row.get('has_freeze_authority')}")
            else:
                print("  (sin fila en `tokens` para esta dirección)")
            print(f"  get_token_safety (cache RPC): {safe}")
            print(
                "\nNota: en exploradores a veces se muestra la freeze *authority* (wallet), "
                "no el mint del token. Para ventas fallidas, usa el mint del trade."
            )
            return

        rows = await db.audit_token_safety_for_traded_mints(limit=args.limit)
        print(f"\nMints en trades (hasta {args.limit}), con join a `tokens`:\n")
        print(f"{'address':<44} {'tr#':>4} {'freeze?':>8} {'program':<45} symbol")
        for r in rows:
            hf = r.get("has_freeze_authority")
            hf_s = "?" if hf is None else str(int(hf))
            prog = (r.get("token_program") or "")[:44]
            sym = (r.get("symbol") or "")[:20]
            addr = (r.get("address") or "")[:44]
            print(f"{addr:<44} {r.get('trade_count', 0):>4} {hf_s:>8} {prog:<45} {sym}")
        print(f"\nTotal filas: {len(rows)}")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
