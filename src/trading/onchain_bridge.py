"""
Puente entre el simulador (decisiones Kelly/EVS) y ejecución real Jupiter+Solana.

Solo actúa si TRADING_MODE=live, LIVE_TRADING_ENABLED y BOT_ONCHAIN_EXECUTION=1.
No modifica umbrales ni tamaños: usa position_size_usd del simulador y lo convierte a SOL.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient

from config.settings import SETTINGS


def sell_error_indicates_no_tokens_to_sell(message: Optional[str]) -> bool:
    """
    Saldo SPL 0: no hay posición on-chain que vender (p. ej. cierre manual desde la cartera).
    No reintentar: permite que el simulador cierre el trade en DB al instante.
    """
    if not message:
        return False
    m = message.lower()
    if "saldo spl 0" in m and "nada que vender" in m:
        return True
    if "nothing to sell" in m:
        return True
    return False


def sell_error_indicates_frozen_token_account(message: Optional[str]) -> bool:
    """
    Detecta fallos de venta por cuenta SPL congelada (Token / Token-2022).
    Evita reintentos inútiles cuando el ATA está frozen (p. ej. custom program error 0x11).
    """
    if not message:
        return False
    m = message.lower()
    if "account is frozen" in m:
        return True
    if "custom program error: 0x11" in m:
        return True
    return False


from src.data_sources.dexscreener import DexScreenerClient
from src.execution.config import get_execution_config
from src.execution import ExecutionEngineV1Jupiter
from src.execution.constants import WSOL_MINT
from src.execution.tx_fee import get_transaction_fee_lamports
from src.execution.wallet import load_keypair_from_env


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


@dataclass
class OnchainBuyResult:
    ok: bool
    tx_signature: Optional[str] = None
    error: Optional[str] = None
    sol_amount: float = 0.0
    fee_lamports: Optional[int] = None


@dataclass
class OnchainSellResult:
    ok: bool
    tx_signature: Optional[str] = None
    error: Optional[str] = None
    fee_lamports: Optional[int] = None
    attempts: int = 0


class BotOnchainBridge:
    """Ejecuta compra/venta en cadena alineada con el simulador."""

    def __init__(self):
        self._sol_usd_cache: Optional[float] = None
        self._sol_usd_cached_at: Optional[float] = None

    @staticmethod
    def _parse_mint_has_freeze_authority(data: bytes) -> Optional[bool]:
        """
        Parse minimal Mint layout (SPL Token / Token-2022) to detect freeze authority.
        Retorna None si el buffer es demasiado corto.
        """
        if not data or len(data) < 82:
            return None
        # offset 46: freeze_authority_option u32 LE (Mint layout base)
        opt = int.from_bytes(data[46:50], "little", signed=False)
        return opt != 0

    @staticmethod
    def _account_data_to_raw_bytes(account_value: Any) -> Optional[bytes]:
        """
        Normaliza `Account.data` según versión de solana-py/solders.

        - Respuesta nueva: `data` es `bytes` / `memoryview` (datos crudos del mint).
        - Respuesta JSON clásica: `data == [base64_str, "base64"]`.
        """
        data = getattr(account_value, "data", None)
        if data is None:
            return None
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, memoryview):
            return bytes(data)
        if isinstance(data, (list, tuple)) and len(data) >= 1:
            chunk = data[0]
            if isinstance(chunk, (bytes, bytearray)):
                return bytes(chunk)
            if isinstance(chunk, str) and chunk.strip():
                try:
                    return base64.b64decode(chunk)
                except Exception:
                    return None
        return None

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

    def min_reserve_usd(self) -> float:
        return _env_float(
            "MIN_WALLET_SOL_RESERVE_USD",
            float(SETTINGS.get("min_wallet_sol_reserve_usd", 10.0)),
        )

    async def wallet_available_usd_for_new_buy(self, dex: DexScreenerClient) -> tuple[Optional[float], str]:
        """
        Capital disponible real (USD aprox.) para una nueva compra on-chain, respetando la
        reserva mínima de gas configurada.
        """
        bal_sol, err = await self._wallet_balance_sol()
        if bal_sol is None:
            return None, err or "balance desconocido"
        sol_usd = await self.fetch_sol_usd(dex)
        if sol_usd is None or sol_usd <= 0:
            return None, "precio SOL no disponible"
        reserve_sol = self.min_reserve_usd() / sol_usd
        free_sol = max(0.0, float(bal_sol) - float(reserve_sol))
        return free_sol * sol_usd, ""

    def sell_retry_timeout_s(self) -> float:
        return _env_float("SELL_RETRY_TIMEOUT_S", float(SETTINGS.get("sell_retry_timeout_s", 600.0)))

    def sell_retry_interval_s(self) -> float:
        return _env_float("SELL_RETRY_INTERVAL_S", float(SETTINGS.get("sell_retry_interval_s", 15.0)))

    async def fetch_sol_usd(self, dex: DexScreenerClient) -> Optional[float]:
        ttl_s = float(SETTINGS.get("sol_price_cache_ttl_seconds", 90.0))
        now = time.monotonic()

        if (
            self._sol_usd_cache is not None
            and self._sol_usd_cached_at is not None
            and (now - self._sol_usd_cached_at) <= ttl_s
        ):
            return float(self._sol_usd_cache)

        prices = await dex.get_prices_batch([WSOL_MINT])
        p = float(prices.get(WSOL_MINT, 0) or 0)
        if p > 0:
            self._sol_usd_cache = p
            self._sol_usd_cached_at = now
            return p

        # Si Dex está rate-limited/no responde, intenta usar caché aunque esté algo vieja.
        stale_ok_s = float(SETTINGS.get("sol_price_cache_stale_ok_seconds", 300.0))
        if (
            self._sol_usd_cache is not None
            and self._sol_usd_cached_at is not None
            and (now - self._sol_usd_cached_at) <= stale_ok_s
        ):
            logger.warning(
                "Precio SOL (DexScreener) no disponible; usando caché stale (age={:.1f}s) USD={:.2f}",
                now - self._sol_usd_cached_at,
                float(self._sol_usd_cache),
            )
            return float(self._sol_usd_cache)

        fail_closed = bool(SETTINGS.get("sol_price_fail_closed_live", True))
        if fail_closed:
            logger.error("Precio SOL (DexScreener) no disponible y sin caché válida; abortando (fail-closed).")
            return None

        fb = float(SETTINGS.get("sol_price_usd_fallback", 140.0))
        logger.warning("Precio SOL (DexScreener) no disponible; fallback USD={}", fb)
        return fb

    async def _wallet_balance_sol(self) -> tuple[Optional[float], Optional[str]]:
        cfg = get_execution_config()
        rpc_url = (cfg.get("rpc_url") or "").strip()
        if not rpc_url:
            return None, "sin RPC"
        kp = load_keypair_from_env()
        if kp is None:
            return None, "sin PRIVATE_KEY"
        owner = kp.pubkey()
        to = float(cfg.get("solana_rpc_timeout_s", 30.0))
        client = AsyncClient(rpc_url, timeout=to)
        try:
            r = await client.get_balance(owner)
            lam = int(r.value or 0)
            return lam / 1e9, None
        finally:
            await client.close()

    async def mint_has_freeze_authority(self, token_mint: str) -> tuple[Optional[bool], Optional[str], str]:
        """
        Lee el mint account por RPC y devuelve:
        (has_freeze_authority, token_program_owner, error_str)
        """
        cfg = get_execution_config()
        rpc_url = (cfg.get("rpc_url") or "").strip()
        if not rpc_url:
            return None, None, "sin RPC"
        client = AsyncClient(rpc_url, timeout=float(cfg.get("solana_rpc_timeout_s", 30.0)))
        try:
            pk = Pubkey.from_string(token_mint)
            r = await client.get_account_info(pk, encoding="base64")
            v = getattr(r, "value", None)
            if v is None:
                return None, None, "mint no encontrado"
            owner = str(getattr(v, "owner", "") or "")
            raw = self._account_data_to_raw_bytes(v)
            if raw is None:
                return None, owner, "mint data formato desconocido"
            if len(raw) == 0:
                return None, owner, "mint data vacío"
            has = self._parse_mint_has_freeze_authority(raw)
            if has is None:
                return None, owner, "mint data demasiado corto"
            return bool(has), owner, ""
        except Exception as e:
            return None, None, str(e)
        finally:
            await client.close()

    async def wallet_has_reserve_for_buy(self, planned_buy_sol: float, dex: DexScreenerClient) -> tuple[bool, str]:
        """
        Exige balance nativo >= compra planificada + reserva (equiv. USD en SOL).
        Evita quedarse sin gas para la venta posterior.
        """
        bal_sol, err = await self._wallet_balance_sol()
        if bal_sol is None:
            return False, err or "balance desconocido"
        sol_usd = await self.fetch_sol_usd(dex)
        if sol_usd is None or sol_usd <= 0:
            return False, "precio SOL no disponible para comprobar reserva"
        reserve_sol = self.min_reserve_usd() / sol_usd
        need = planned_buy_sol + reserve_sol
        if bal_sol < need:
            return (
                False,
                f"SOL insuficiente: balance≈{bal_sol:.6f}, necesitas compra≈{planned_buy_sol:.6f} "
                f"+ reserva≈{reserve_sol:.6f} (~{self.min_reserve_usd():.0f} USD gas)",
            )
        return True, ""

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
    ) -> OnchainBuyResult:
        amount_sol = await self.compute_buy_sol_amount(position_usd, dex)
        if amount_sol is None or amount_sol <= 0:
            return OnchainBuyResult(False, error="monto SOL inválido o precio SOL no disponible")
        ok_r, msg_r = await self.wallet_has_reserve_for_buy(amount_sol, dex)
        if not ok_r:
            return OnchainBuyResult(False, error=msg_r)

        eng = ExecutionEngineV1Jupiter()
        if not eng.is_live_ready():
            return OnchainBuyResult(False, error="motor live no listo (RPC / clave / flags)")
        res = await eng.execute_trade(token_mint, amount_sol, db=db)
        if not (res.success and res.tx_signature):
            return OnchainBuyResult(
                False,
                tx_signature=res.tx_signature,
                error=res.error or res.status,
                sol_amount=amount_sol,
            )
        cfg = get_execution_config()
        fee = await get_transaction_fee_lamports(
            (cfg.get("rpc_url") or "").strip(),
            res.tx_signature,
            timeout_s=float(cfg.get("solana_rpc_timeout_s", 30.0)),
        )
        return OnchainBuyResult(
            True,
            tx_signature=res.tx_signature,
            sol_amount=amount_sol,
            fee_lamports=fee,
        )

    async def execute_sell_for_close(self, token_mint: str, db) -> OnchainSellResult:
        """Vende todo el SPL del mint; reintenta hasta sell_retry_timeout_s."""
        eng0 = ExecutionEngineV1Jupiter()
        if not eng0.is_live_ready():
            return OnchainSellResult(False, error="motor live no listo (RPC / clave / flags)", attempts=0)

        deadline = time.monotonic() + self.sell_retry_timeout_s()
        interval = self.sell_retry_interval_s()
        last_err: Optional[str] = None
        attempts = 0
        cfg = get_execution_config()
        rpc_url = (cfg.get("rpc_url") or "").strip()
        rpc_to = float(cfg.get("solana_rpc_timeout_s", 30.0))

        while time.monotonic() < deadline:
            attempts += 1
            eng = ExecutionEngineV1Jupiter()
            res = await eng.execute_sell_all(token_mint, db=db)
            if res.success and res.tx_signature:
                fee = await get_transaction_fee_lamports(
                    rpc_url,
                    res.tx_signature,
                    timeout_s=rpc_to,
                )
                return OnchainSellResult(
                    True,
                    tx_signature=res.tx_signature,
                    fee_lamports=fee,
                    attempts=attempts,
                )
            last_err = res.error or res.status
            if sell_error_indicates_frozen_token_account(last_err):
                logger.error(
                    "Venta on-chain abortada (cuenta/token congelada, sin reintentos): {}",
                    last_err,
                )
                return OnchainSellResult(
                    False,
                    error=last_err,
                    attempts=attempts,
                )
            if sell_error_indicates_no_tokens_to_sell(last_err):
                logger.info(
                    "Venta on-chain omitida (sin SPL del mint en wallet; posible venta manual). "
                    "Se considera posición ya cerrada en cadena: {}",
                    last_err,
                )
                return OnchainSellResult(True, attempts=attempts)
            logger.warning(
                "Venta on-chain intento {} falló: {}; reintento en {:.0f}s",
                attempts,
                last_err,
                interval,
            )
            await asyncio.sleep(interval)

        return OnchainSellResult(
            False,
            error=last_err or "timeout reintentos venta",
            attempts=attempts,
        )
