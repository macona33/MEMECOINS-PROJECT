"""Balances SPL (raw units) para medir amount_out real."""

from typing import Optional

from loguru import logger
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TokenAccountOpts


async def get_spl_token_raw_balance(
    rpc: AsyncClient,
    owner: Pubkey,
    mint: Pubkey,
) -> int:
    """Suma de `amount` en unidades mínimas del token para todas las cuentas del owner y mint."""
    try:
        resp = await rpc.get_token_accounts_by_owner_json_parsed(
            owner,
            TokenAccountOpts(mint=mint),
        )
    except Exception as e:
        logger.warning(f"get_token_accounts_by_owner_json_parsed falló: {e}")
        return 0

    if resp.value is None:
        return 0

    total = 0
    for acc in resp.value:
        try:
            parsed = acc.account.data.parsed
            info = parsed.get("info", {})
            ta = info.get("tokenAmount", {})
            amt = ta.get("amount")
            if amt is not None:
                total += int(amt)
        except Exception:
            continue
    return total
