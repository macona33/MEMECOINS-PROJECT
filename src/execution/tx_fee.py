"""Lectura de comisión (fee) de una transacción confirmada vía RPC."""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.signature import Signature


async def get_transaction_fee_lamports(
    rpc_url: str,
    signature: str,
    *,
    timeout_s: float = 30.0,
    max_attempts: int = 8,
    delay_s: float = 0.75,
) -> Optional[int]:
    """Devuelve la fee en lamports según meta de la tx, o None si no disponible."""
    if not rpc_url or not signature or signature == "SIMULATED":
        return None
    sig = Signature.from_string(signature)
    client = AsyncClient(rpc_url, timeout=timeout_s)
    try:
        for attempt in range(max_attempts):
            try:
                resp = await client.get_transaction(
                    sig,
                    commitment=Confirmed,
                    max_supported_transaction_version=0,
                )
                val = resp.value
                if val is None or val.transaction is None:
                    await asyncio.sleep(delay_s)
                    continue
                meta = getattr(val.transaction, "meta", None)
                if meta is None:
                    return None
                fee = getattr(meta, "fee", None)
                if fee is not None:
                    return int(fee)
            except Exception as e:
                logger.debug("get_transaction fee intento {}: {}", attempt + 1, e)
                await asyncio.sleep(delay_s)
        return None
    finally:
        await client.close()
