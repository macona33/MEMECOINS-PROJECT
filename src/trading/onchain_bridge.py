"""
Puente entre el simulador (decisiones Kelly/EVS) y ejecución real Jupiter+Solana.

Solo actúa si TRADING_MODE=live, LIVE_TRADING_ENABLED y BOT_ONCHAIN_EXECUTION=1.
No modifica umbrales ni tamaños: usa position_size_usd del simulador y lo convierte a SOL.
"""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

from config.settings import SETTINGS
from src.data_sources.dexscreener import DexScreenerClient
from src.execution.config import get_execution_config
from src.execution import ExecutionEngineV1Jupiter
from src.execution.constants import WSOL_MINT


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


class BotOnchainBridge:
    """Ejecuta compra/venta en cadena alineada con el simulador."""

    def is_active(self) -> bool:
        cfg = get_execution_config()
        if cfg.get("trading_mode") != "live" or not cfg.get("live_trading_enabled"):
            return False
        if not (cfg.get("rpc_url") or "").strip():
            return False
        env = os.getenv("BOT_ONCHAIN_EXECUTION")
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return bool(SETTINGS.get("bot_onchain_execution", False))

    def max_sol_per_trade(self) -> float:
        return _env_float("MAX_ONCHAIN_SOL_PER_TRADE", float(SETTINGS.get("max_onchain_sol_per_trade", 0.06)))

    def min_sol_per_trade(self) -> float:
        return _env_float("MIN_ONCHAIN_SOL_PER_TRADE", float(SETTINGS.get("min_onchain_sol_per_trade", 0.001)))

    async def fetch_sol_usd(self, dex: DexScreenerClient) -> Optional[float]:
        prices = await dex.get_prices_batch([WSOL_MINT])
        p = float(prices.get(WSOL_MINT, 0) or 0)
        if p > 0:
            return p
        fb = float(SETTINGS.get("sol_price_usd_fallback", 140.0))
        logger.warning("Precio SOL (DexScreener) no disponible; fallback USD={}", fb)
        return fb

    async def compute_buy_sol_amount(
        self,
        position_usd: float,
        dex: DexScreenerClient,
    ) -> Optional[float]:
        if position_usd <= 0:
            return None
        sol_usd = await self.fetch_sol_usd(dex)
        if sol_usd is None or sol_usd <= 0:
            return None
        raw = position_usd / sol_usd
        cap = self.max_sol_per_trade()
        floor = self.min_sol_per_trade()
        amount = min(raw, cap)
        if amount < floor:
            logger.warning(
                "Monto SOL calculado {:.6f} < mínimo {:.6f}; se usa el mínimo",
                amount,
                floor,
            )
            amount = floor
        return amount

    async def execute_buy_for_open(
        self,
        token_mint: str,
        position_usd: float,
        dex: DexScreenerClient,
        db,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Compra on-chain acorde a position_usd (capado).
        Retorna (ok, tx_signature o None, mensaje_error).
        """
        amount_sol = await self.compute_buy_sol_amount(position_usd, dex)
        if amount_sol is None or amount_sol <= 0:
            return False, None, "monto SOL inválido o precio SOL no disponible"
        eng = ExecutionEngineV1Jupiter()
        if not eng.is_live_ready():
            return False, None, "motor live no listo (RPC / clave / flags)"
        res = await eng.execute_trade(token_mint, amount_sol, db=db)
        if res.success and res.tx_signature:
            return True, res.tx_signature, None
        return False, res.tx_signature, res.error or res.status

    async def execute_sell_for_close(self, token_mint: str, db) -> tuple[bool, Optional[str], Optional[str]]:
        """Vende todo el SPL del mint (cierra posición en wallet)."""
        eng = ExecutionEngineV1Jupiter()
        if not eng.is_live_ready():
            return False, None, "motor live no listo (RPC / clave / flags)"
        res = await eng.execute_sell_all(token_mint, db=db)
        if res.success and res.tx_signature:
            return True, res.tx_signature, None
        return False, res.tx_signature, res.error or res.status
