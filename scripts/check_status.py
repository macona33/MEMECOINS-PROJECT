"""
Script para verificar estado actual del sistema.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import DatabaseManager


async def check_status():
    db = DatabaseManager()
    await db.connect()
    
    try:
        tokens = await db.get_recent_tokens(hours=1)
        print(f"\n=== TOKENS DETECTADOS (ultima hora): {len(tokens)} ===")
        for t in tokens[:10]:
            symbol = t.get("symbol", "N/A")
            liq = t.get("liquidity_usd", 0) or 0
            print(f"  - {symbol}: Liquidez ${liq:,.0f}")
        
        active = await db.get_active_trades()
        print(f"\n=== TRADES ACTIVOS: {len(active)} ===")
        for t in active:
            addr = t.get("token_address", "")[:12]
            entry = t.get("entry_price", 0)
            size = t.get("position_size_usd", 0)
            print(f"  - {addr}... Entry: ${entry:.8f}, Size: ${size:.2f}")
        
        top = await db.get_top_ranked_tokens(limit=5)
        print(f"\n=== TOP TOKENS POR EVS ===")
        for t in top:
            symbol = t.get("symbol", "N/A")
            evs = t.get("evs_adj", 0) or 0
            p_rug = t.get("p_rug", 0) or 0
            print(f"  - {symbol}: EVS_adj={evs:.4f}, P_rug={p_rug:.1%}")
        
        print("\n=== SISTEMA FUNCIONANDO CORRECTAMENTE ===\n")
        
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(check_status())
