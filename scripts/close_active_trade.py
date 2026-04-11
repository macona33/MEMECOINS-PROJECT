"""
Cierra un trade activo por dirección de mint (útil si el precio DexScreener falla).

Uso:
  python scripts/close_active_trade.py 9d4JKN9bfuWRZ974jkydf2JVecWgjzKL65oXjUpuUQjH
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.storage import DatabaseManager
from src.data_sources import DexScreenerClient
from src.trading import TradeSimulator


async def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python scripts/close_active_trade.py <TOKEN_ADDRESS>")
        sys.exit(1)
    token_address = sys.argv[1].strip()

    db = DatabaseManager()
    dex = DexScreenerClient()
    sim = TradeSimulator(db=db, dex_client=dex)
    await db.connect()
    try:
        await sim.load_active_trades()
        trade = sim.get_trade(token_address)
        if not trade:
            logger.error(f"No hay trade activo para {token_address}")
            sys.exit(2)
        prices = await dex.get_prices_batch([token_address])
        px = prices.get(token_address, 0) or 0
        if px <= 0:
            px = trade.current_price or trade.entry_price
            logger.info(f"Sin precio DexScreener; cierre a último conocido / entry: {px}")
        closed = await sim.close_trade(token_address, px, "manual_close")
        if closed:
            logger.info(f"Trade cerrado: {closed.symbol} reason=manual_close")
        else:
            logger.error("close_trade devolvió None")
            sys.exit(3)
    finally:
        await dex.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
