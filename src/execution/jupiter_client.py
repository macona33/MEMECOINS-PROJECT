"""Cliente async Jupiter Swap API (quote + swap; host configurable)."""

from typing import Any, Dict, Optional

import httpx
from loguru import logger


class JupiterClientError(Exception):
    """Error de API Jupiter o validación de quote."""


class JupiterV6Client:
    def __init__(self, api_base: str, quote_timeout: float = 20.0, swap_timeout: float = 30.0):
        self._base = api_base.rstrip("/")
        self._quote_timeout = quote_timeout
        self._swap_timeout = swap_timeout

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount_lamports: int,
        slippage_bps: int,
    ) -> Dict[str, Any]:
        url = f"{self._base}/quote"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(int(amount_lamports)),
            "slippageBps": str(int(slippage_bps)),
        }
        async with httpx.AsyncClient(timeout=self._quote_timeout) as client:
            r = await client.get(url, params=params)
        if r.status_code != 200:
            raise JupiterClientError(f"Quote HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            raise JupiterClientError(str(data.get("error")))
        self._validate_quote(data)
        return data

    def _validate_quote(self, data: Dict[str, Any]) -> None:
        routes = data.get("routePlan")
        if routes is None or (isinstance(routes, list) and len(routes) == 0):
            raise JupiterClientError("routePlan vacío o ausente")

        out = data.get("outAmount")
        if out is None or int(out) <= 0:
            raise JupiterClientError("outAmount inválido")

        pip = data.get("priceImpactPct")
        if pip is None:
            raise JupiterClientError("priceImpactPct ausente")
        try:
            float(pip)
        except (TypeError, ValueError) as e:
            raise JupiterClientError("priceImpactPct no parseable") from e

    async def get_swap_transaction_b64(
        self,
        quote_response: Dict[str, Any],
        user_public_key: str,
    ) -> str:
        url = f"{self._base}/swap"
        body = {
            "quoteResponse": quote_response,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True,
        }
        async with httpx.AsyncClient(timeout=self._swap_timeout) as client:
            r = await client.post(url, json=body)
        if r.status_code != 200:
            raise JupiterClientError(f"Swap HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            raise JupiterClientError(str(data.get("error")))
        b64 = data.get("swapTransaction")
        if not b64 or not isinstance(b64, str):
            raise JupiterClientError("swapTransaction ausente o inválido")
        return b64
