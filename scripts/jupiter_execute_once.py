"""
Prueba controlada del ExecutionEngine_v1_Jupiter.

Por defecto usa modo paper (TRADING_MODE=paper o sin LIVE_TRADING_ENABLED).
No envía transacciones reales salvo TRADING_MODE=live y LIVE_TRADING_ENABLED=1.

Compra (por defecto): SOL → SPL (--amount-sol).
Venta: SPL → SOL con --sell y --amount-token-raw, o --sell-all (100 % del saldo del mint).
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.storage import DatabaseManager
from src.execution import ExecutionEngineV1Jupiter


async def main():
    parser = argparse.ArgumentParser(description="Jupiter execute_once (FASE 3)")
    parser.add_argument("--mint", type=str, required=True, help="Mint SPL del token (base58)")
    parser.add_argument("--sell", action="store_true", help="Modo venta: token → SOL")
    parser.add_argument(
        "--amount-sol",
        type=float,
        default=0.001,
        help="SOL a gastar en compra (ignorado con --sell)",
    )
    parser.add_argument(
        "--amount-token-raw",
        type=int,
        default=None,
        help="Cantidad en unidades mínimas del token a vender (con --sell, sin --sell-all)",
    )
    parser.add_argument(
        "--sell-all",
        action="store_true",
        help="Vender todo el saldo SPL del --mint (requiere RPC + PRIVATE_KEY para leer cartera)",
    )
    parser.add_argument("--dry-quote-only", action="store_true", help="Solo quote (sin persistir ejecución)")
    parser.add_argument("--no-db", action="store_true", help="No escribir execution_logs")
    args = parser.parse_args()

    sell_mode = args.sell or args.sell_all
    if args.sell_all and args.amount_token_raw is not None:
        parser.error("No uses --sell-all junto con --amount-token-raw")
    if args.sell and args.amount_token_raw is None and not args.sell_all:
        parser.error("Con --sell indica --amount-token-raw o usa --sell-all")
    if not sell_mode and args.amount_token_raw is not None:
        parser.error("--amount-token-raw solo aplica en venta (--sell o --sell-all)")

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    db = None
    if not args.no_db and not args.dry_quote_only:
        db = DatabaseManager()
        await db.connect()

    try:
        eng = ExecutionEngineV1Jupiter()
        logger.info(f"live_ready={eng.is_live_ready()}")
        if args.dry_quote_only:
            from src.execution.config import get_execution_config
            from src.execution.jupiter_client import JupiterV6Client
            from src.execution.constants import WSOL_MINT

            cfg = get_execution_config()
            j = JupiterV6Client(
                cfg["jupiter_api_base"],
                quote_timeout=cfg["jupiter_quote_timeout_s"],
            )
            slip = cfg["execution_slippage_bps"]
            if sell_mode:
                if args.sell_all:
                    bal, berr = await eng.get_wallet_token_balance_raw(args.mint)
                    if berr:
                        logger.error("sell-all quote: {}", berr)
                        return
                    if bal <= 0:
                        logger.error("sell-all quote: saldo SPL 0 para este mint")
                        return
                    raw_amt = bal
                else:
                    raw_amt = args.amount_token_raw
                assert raw_amt is not None and raw_amt > 0
                q = await j.get_quote(
                    args.mint,
                    WSOL_MINT,
                    raw_amt,
                    slip,
                )
                logger.info(
                    f"Quote SELL OK amount_token_raw={raw_amt} "
                    f"outAmount(lamports SOL)={q.get('outAmount')} "
                    f"priceImpactPct={q.get('priceImpactPct')}"
                )
            else:
                lamports = int(args.amount_sol * 1_000_000_000)
                q = await j.get_quote(
                    WSOL_MINT,
                    args.mint,
                    lamports,
                    slip,
                )
                logger.info(f"Quote BUY OK outAmount={q.get('outAmount')} priceImpactPct={q.get('priceImpactPct')}")
            return

        if args.sell_all:
            res = await eng.execute_sell_all(args.mint, db=db)
        elif args.sell:
            assert args.amount_token_raw is not None
            res = await eng.execute_sell(args.mint, args.amount_token_raw, db=db)
        else:
            res = await eng.execute_trade(args.mint, args.amount_sol, db=db)
        logger.info(
            f"Result: success={res.success} status={res.status} sig={res.tx_signature} "
            f"slip_real={res.slippage_real} ms={res.execution_time_ms} err={res.error}"
        )
    finally:
        if db:
            await db.close()


if __name__ == "__main__":
    asyncio.run(main())
