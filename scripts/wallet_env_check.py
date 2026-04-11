"""
Comprueba que .env apunta a la wallet correcta: pubkey + balance en el RPC configurado.
Uso (desde la raíz del repo): python scripts/wallet_env_check.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger
from solana.rpc.async_api import AsyncClient


async def main() -> None:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    from src.execution.wallet import load_keypair_from_env

    kp = load_keypair_from_env()
    if kp is None:
        logger.error("PRIVATE_KEY no definida o vacía en .env")
        sys.exit(1)

    import os

    rpc = (os.getenv("SOLANA_RPC_URL") or "").strip()
    if not rpc:
        logger.error("SOLANA_RPC_URL vacío")
        sys.exit(1)

    owner = kp.pubkey()
    logger.info("Pubkey derivada de PRIVATE_KEY: {}", owner)

    async with AsyncClient(rpc, timeout=30.0) as client:
        bal = await client.get_balance(owner)
        lam = bal.value or 0
        logger.info("Balance en este RPC: {} lamports (~{:.9f} SOL)", lam, lam / 1e9)

    logger.info(
        "Compara la pubkey con Phantom (copiar dirección). Si no coincide, exporta de nuevo la clave de esa cuenta."
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(main())
