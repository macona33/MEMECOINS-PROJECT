"""
ExecutionEngine_v1_Jupiter — FASE 3.
Compra SOL→SPL y venta SPL→SOL vía Jupiter Swap API + firma + RPC Solana.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from loguru import logger
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts

from src.execution.balances import get_spl_token_raw_balance
from src.execution.config import get_execution_config
from src.execution.constants import WSOL_MINT
from src.execution.jupiter_client import JupiterClientError, JupiterV6Client
from src.execution.solana_tx import sign_jupiter_versioned_transaction
from src.execution.wallet import load_keypair_from_env


@dataclass
class ExecutionResult:
    success: bool
    status: str
    tx_signature: Optional[str]
    slippage_real: Optional[float]
    slippage_expected_bps: int
    expected_out_raw: Optional[int]
    real_out_raw: Optional[int]
    price_impact_pct: Optional[float]
    error: Optional[str]
    execution_time_ms: int

    def to_log_row(self, token_mint: str, amount_in_lamports: int, mode: str) -> Dict[str, Any]:
        return {
            "token_mint": token_mint,
            "amount_in_lamports": amount_in_lamports,
            "expected_out_raw": self.expected_out_raw,
            "real_out_raw": self.real_out_raw,
            "slippage_expected_bps": self.slippage_expected_bps,
            "slippage_real": self.slippage_real,
            "price_impact_pct": self.price_impact_pct,
            "tx_signature": self.tx_signature,
            "status": self.status,
            "error_message": self.error,
            "execution_time_ms": self.execution_time_ms,
            "mode": mode,
        }


async def _wait_signature_confirmed(
    rpc: AsyncClient,
    sig_str: str,
    timeout_s: float,
) -> tuple[Optional[bool], Optional[str]]:
    """True confirmado OK, False confirmado con error on-chain, None timeout."""
    from solders.signature import Signature

    sig = Signature.from_string(sig_str)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        resp = await rpc.get_signature_statuses([sig])
        if resp.value and len(resp.value) > 0:
            st = resp.value[0]
            if st is None:
                await asyncio.sleep(0.45)
                continue
            if st.err:
                return False, str(st.err)
            if st.confirmation_status is not None:
                return True, None
        await asyncio.sleep(0.45)
    return None, "confirm_timeout"


class ExecutionEngineV1Jupiter:
    """
    Motor de ejecución Jupiter + Solana (compra SOL→SPL y venta SPL→SOL).
    Protecciones: cooldown, reintentos RPC, parada por fallos consecutivos de envío/confirmación.
    """

    def __init__(self) -> None:
        self._cfg = get_execution_config()
        self._jupiter = JupiterV6Client(
            self._cfg["jupiter_api_base"],
            quote_timeout=self._cfg["jupiter_quote_timeout_s"],
            swap_timeout=self._cfg["jupiter_swap_timeout_s"],
        )
        self._last_exec_mono: float = 0.0
        self._consecutive_failures = 0
        self._disabled = False
        self._lock = asyncio.Lock()

    def is_live_ready(self) -> bool:
        cfg = get_execution_config()
        return (
            cfg["trading_mode"] == "live"
            and cfg["live_trading_enabled"]
            and bool((cfg.get("rpc_url") or "").strip())
        )

    def _parse_price_impact(self, quote: Dict[str, Any]) -> float:
        pip = quote.get("priceImpactPct")
        return float(pip) if pip is not None else 0.0

    async def execute_trade(
        self,
        token_address: str,
        amount_sol: float,
        db: Optional[Any] = None,
    ) -> ExecutionResult:
        """Compra: SOL (wrapped) → SPL del mint indicado."""
        lamports = int(amount_sol * 1_000_000_000)
        token_pk = Pubkey.from_string(token_address)

        async def live_precheck(rpc: AsyncClient, owner: Pubkey) -> Optional[str]:
            bal0 = await rpc.get_balance(owner)
            lam0 = bal0.value or 0
            logger.info(
                "LIVE signer={} balance_lamports={} (~{:.6f} SOL)",
                owner,
                lam0,
                lam0 / 1e9,
            )
            if lam0 < lamports + 500_000:
                logger.warning(
                    "Balance bajo para compra: swap_lamports={} balance_lamports={}",
                    lamports,
                    lam0,
                )
            return None

        async def measure_out(rpc: AsyncClient, owner: Pubkey) -> int:
            return await get_spl_token_raw_balance(rpc, owner, token_pk)

        return await self._execute_swap(
            input_mint=WSOL_MINT,
            output_mint=token_address,
            amount_in_raw=lamports,
            persist_token_mint=token_address,
            persist_amount_in=lamports,
            invalid_amount_msg="amount_sol inválido",
            db=db,
            side_label="BUY",
            live_precheck=live_precheck,
            measure_output=measure_out,
        )

    async def execute_sell(
        self,
        token_address: str,
        amount_token_raw: int,
        db: Optional[Any] = None,
    ) -> ExecutionResult:
        """Venta: cantidad en unidades mínimas del SPL → SOL (unwrap vía Jupiter)."""
        token_pk = Pubkey.from_string(token_address)

        async def live_precheck(rpc: AsyncClient, owner: Pubkey) -> Optional[str]:
            bal0 = await rpc.get_balance(owner)
            lam0 = bal0.value or 0
            spl0 = await get_spl_token_raw_balance(rpc, owner, token_pk)
            logger.info(
                "LIVE sell signer={} sol_lamports={} (~{:.6f} SOL) spl_raw={}",
                owner,
                lam0,
                lam0 / 1e9,
                spl0,
            )
            if spl0 < amount_token_raw:
                return f"saldo SPL insuficiente (tienes {spl0}, pides {amount_token_raw})"
            if lam0 < 80_000:
                return "SOL insuficiente para comisión de red (~0.00008 SOL mínimo aprox.)"
            return None

        async def measure_out(rpc: AsyncClient, owner: Pubkey) -> int:
            r = await rpc.get_balance(owner)
            return int(r.value or 0)

        return await self._execute_swap(
            input_mint=token_address,
            output_mint=WSOL_MINT,
            amount_in_raw=amount_token_raw,
            persist_token_mint=token_address,
            persist_amount_in=amount_token_raw,
            invalid_amount_msg="amount_token_raw inválido",
            db=db,
            side_label="SELL",
            live_precheck=live_precheck,
            measure_output=measure_out,
        )

    async def get_wallet_token_balance_raw(self, token_mint: str) -> tuple[int, Optional[str]]:
        """
        Saldo SPL (unidades mínimas) del mint en la wallet de PRIVATE_KEY.
        (0, None) si la cuenta de token no existe o saldo 0.
        """
        cfg = get_execution_config()
        rpc_url = (cfg.get("rpc_url") or "").strip()
        if not rpc_url:
            return 0, "consulta de saldo requiere SOLANA_RPC_URL"
        kp = load_keypair_from_env()
        if kp is None:
            return 0, "consulta de saldo requiere PRIVATE_KEY"
        owner = kp.pubkey()
        mint_pk = Pubkey.from_string(token_mint)
        rpc = AsyncClient(rpc_url, timeout=float(cfg["solana_rpc_timeout_s"]))
        try:
            bal = await get_spl_token_raw_balance(rpc, owner, mint_pk)
            return int(bal), None
        finally:
            await rpc.close()

    async def execute_sell_all(self, token_address: str, db: Optional[Any] = None) -> ExecutionResult:
        """Vende todo el SPL disponible del mint en la wallet (cierra posición en token)."""
        t0 = time.perf_counter()
        cfg = get_execution_config()
        slippage_bps = int(cfg["execution_slippage_bps"])
        bal, err = await self.get_wallet_token_balance_raw(token_address)
        ms = int((time.perf_counter() - t0) * 1000)
        if err:
            return ExecutionResult(
                success=False,
                status="REJECTED",
                tx_signature=None,
                slippage_real=None,
                slippage_expected_bps=slippage_bps,
                expected_out_raw=None,
                real_out_raw=None,
                price_impact_pct=None,
                error=err,
                execution_time_ms=ms,
            )
        if bal <= 0:
            return ExecutionResult(
                success=False,
                status="REJECTED",
                tx_signature=None,
                slippage_real=None,
                slippage_expected_bps=slippage_bps,
                expected_out_raw=None,
                real_out_raw=None,
                price_impact_pct=None,
                error="saldo SPL 0 para este mint (nada que vender)",
                execution_time_ms=ms,
            )
        logger.info(
            "sell-all: saldo SPL detectado raw={} mint={}…",
            bal,
            token_address[:8],
        )
        return await self.execute_sell(token_address, bal, db=db)

    async def _execute_swap(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount_in_raw: int,
        persist_token_mint: str,
        persist_amount_in: int,
        invalid_amount_msg: str,
        db: Optional[Any],
        side_label: str,
        live_precheck: Callable[[AsyncClient, Pubkey], Awaitable[Optional[str]]],
        measure_output: Callable[[AsyncClient, Pubkey], Awaitable[int]],
    ) -> ExecutionResult:
        t_start = time.perf_counter()
        cfg = get_execution_config()
        slippage_bps = int(cfg["execution_slippage_bps"])
        max_pi = float(cfg["execution_max_price_impact"])
        mode = cfg["trading_mode"]
        live_ok = cfg["trading_mode"] == "live" and cfg["live_trading_enabled"]

        async with self._lock:
            if self._disabled:
                ms = int((time.perf_counter() - t_start) * 1000)
                return ExecutionResult(
                    success=False,
                    status="DISABLED",
                    tx_signature=None,
                    slippage_real=None,
                    slippage_expected_bps=slippage_bps,
                    expected_out_raw=None,
                    real_out_raw=None,
                    price_impact_pct=None,
                    error="Motor deshabilitado por fallos consecutivos",
                    execution_time_ms=ms,
                )

            cd = float(cfg["execution_trade_cooldown_s"])
            now = time.monotonic()
            wait_s = (self._last_exec_mono + cd) - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)

            if amount_in_raw <= 0:
                ms = int((time.perf_counter() - t_start) * 1000)
                return ExecutionResult(
                    success=False,
                    status="REJECTED",
                    tx_signature=None,
                    slippage_real=None,
                    slippage_expected_bps=slippage_bps,
                    expected_out_raw=None,
                    real_out_raw=None,
                    price_impact_pct=None,
                    error=invalid_amount_msg,
                    execution_time_ms=ms,
                )

            try:
                quote = await self._jupiter.get_quote(
                    input_mint,
                    output_mint,
                    amount_in_raw,
                    slippage_bps,
                )
            except JupiterClientError as e:
                self._last_exec_mono = time.monotonic()
                ms = int((time.perf_counter() - t_start) * 1000)
                res = ExecutionResult(
                    success=False,
                    status="REJECTED",
                    tx_signature=None,
                    slippage_real=None,
                    slippage_expected_bps=slippage_bps,
                    expected_out_raw=None,
                    real_out_raw=None,
                    price_impact_pct=None,
                    error=str(e),
                    execution_time_ms=ms,
                )
                await self._persist(db, persist_token_mint, persist_amount_in, mode, res)
                return res

            price_impact = self._parse_price_impact(quote)
            if price_impact > max_pi:
                self._last_exec_mono = time.monotonic()
                ms = int((time.perf_counter() - t_start) * 1000)
                err = f"priceImpactPct {price_impact:.4f} > max {max_pi}"
                res = ExecutionResult(
                    success=False,
                    status="REJECTED",
                    tx_signature=None,
                    slippage_real=None,
                    slippage_expected_bps=slippage_bps,
                    expected_out_raw=int(quote.get("outAmount", 0)),
                    real_out_raw=None,
                    price_impact_pct=price_impact,
                    error=err,
                    execution_time_ms=ms,
                )
                await self._persist(db, persist_token_mint, persist_amount_in, mode, res)
                return res

            expected_out = int(quote["outAmount"])

            if not live_ok:
                slip_pct = float(cfg["paper_slippage_pct"])
                real_est = int(max(0, expected_out * (1.0 - slip_pct)))
                slip_real = (
                    (expected_out - real_est) / expected_out if expected_out > 0 else 0.0
                )
                self._last_exec_mono = time.monotonic()
                self._consecutive_failures = 0
                ms = int((time.perf_counter() - t_start) * 1000)
                res = ExecutionResult(
                    success=True,
                    status="PAPER_SIM",
                    tx_signature="SIMULATED",
                    slippage_real=slip_real,
                    slippage_expected_bps=slippage_bps,
                    expected_out_raw=expected_out,
                    real_out_raw=real_est,
                    price_impact_pct=price_impact,
                    error=None,
                    execution_time_ms=ms,
                )
                logger.info(
                    f"[PAPER {side_label}] token={persist_token_mint[:8]}… "
                    f"expected_out={expected_out} sim_real_out={real_est} "
                    f"slip_real={slip_real:.4f} impact={price_impact:.4f}"
                )
                await self._persist(db, persist_token_mint, persist_amount_in, mode, res)
                return res

            rpc_url = cfg["rpc_url"]
            if not rpc_url:
                ms = int((time.perf_counter() - t_start) * 1000)
                return ExecutionResult(
                    success=False,
                    status="FAILED",
                    tx_signature=None,
                    slippage_real=None,
                    slippage_expected_bps=slippage_bps,
                    expected_out_raw=expected_out,
                    real_out_raw=None,
                    price_impact_pct=price_impact,
                    error="SOLANA_RPC_URL vacío",
                    execution_time_ms=ms,
                )

            keypair = load_keypair_from_env()
            if keypair is None:
                ms = int((time.perf_counter() - t_start) * 1000)
                return ExecutionResult(
                    success=False,
                    status="FAILED",
                    tx_signature=None,
                    slippage_real=None,
                    slippage_expected_bps=slippage_bps,
                    expected_out_raw=expected_out,
                    real_out_raw=None,
                    price_impact_pct=price_impact,
                    error="PRIVATE_KEY no configurada",
                    execution_time_ms=ms,
                )

            owner = keypair.pubkey()
            rpc_timeout = float(cfg["solana_rpc_timeout_s"])
            max_retries = int(cfg["execution_rpc_max_retries"])
            confirm_timeout = float(cfg["execution_confirm_timeout_s"])

            rpc = AsyncClient(rpc_url, timeout=rpc_timeout)
            try:
                chk = await live_precheck(rpc, owner)
                if chk:
                    ms = int((time.perf_counter() - t_start) * 1000)
                    res = ExecutionResult(
                        success=False,
                        status="REJECTED",
                        tx_signature=None,
                        slippage_real=None,
                        slippage_expected_bps=slippage_bps,
                        expected_out_raw=expected_out,
                        real_out_raw=None,
                        price_impact_pct=price_impact,
                        error=chk,
                        execution_time_ms=ms,
                    )
                    await self._persist(db, persist_token_mint, persist_amount_in, mode, res)
                    return res

                pre_out = await measure_output(rpc, owner)

                swap_b64 = await self._jupiter.get_swap_transaction_b64(
                    quote, str(owner)
                )
                _v0 = VersionedTransaction.from_bytes(base64.b64decode(swap_b64))
                _keys = list(_v0.message.account_keys)
                _n_sig = int(_v0.message.header.num_required_signatures)
                if _keys:
                    logger.info(
                        "Jupiter tx: fee_payer={} required_signatures={} fee_payer_es_wallet={}",
                        _keys[0],
                        _n_sig,
                        _keys[0] == owner,
                    )

                signed = sign_jupiter_versioned_transaction(swap_b64, keypair)
                raw_tx = bytes(signed)

                last_exc: Optional[Exception] = None
                sig_str: Optional[str] = None
                for attempt in range(max_retries + 1):
                    try:
                        send_resp = await rpc.send_raw_transaction(
                            raw_tx,
                            TxOpts(skip_preflight=False, preflight_commitment=Confirmed),
                        )
                        sig_str = str(send_resp.value)
                        break
                    except Exception as e:
                        last_exc = e
                        logger.warning(f"send_raw_transaction intento {attempt + 1}: {e}")
                        if attempt >= max_retries:
                            try:
                                sim = await rpc.simulate_transaction(
                                    signed,
                                    sig_verify=False,
                                    replace_recent_blockhash=True,
                                )
                                sv = sim.value
                                if sv.err:
                                    logger.error(
                                        "simulate_transaction (tras fallo): err={} logs={}",
                                        sv.err,
                                        sv.logs,
                                    )
                                elif sv.logs:
                                    logger.info("simulate_transaction logs={}", sv.logs)
                            except Exception as sim_ex:
                                logger.debug("simulate tras fallo send: {}", sim_ex)

                            self._consecutive_failures += 1
                            if self._consecutive_failures >= int(
                                cfg["execution_max_consecutive_failures"]
                            ):
                                self._disabled = True
                            self._last_exec_mono = time.monotonic()
                            ms = int((time.perf_counter() - t_start) * 1000)
                            res = ExecutionResult(
                                success=False,
                                status="FAILED",
                                tx_signature=None,
                                slippage_real=None,
                                slippage_expected_bps=slippage_bps,
                                expected_out_raw=expected_out,
                                real_out_raw=None,
                                price_impact_pct=price_impact,
                                error=str(last_exc),
                                execution_time_ms=ms,
                            )
                            await self._persist(db, persist_token_mint, persist_amount_in, mode, res)
                            return res
                        await asyncio.sleep(0.4 * (attempt + 1))

                assert sig_str is not None
                ok, cerr = await _wait_signature_confirmed(
                    rpc, sig_str, confirm_timeout
                )
                post_out = await measure_output(rpc, owner)
                real_delta = max(0, post_out - pre_out)
                slip_real = (
                    (expected_out - real_delta) / expected_out if expected_out > 0 else 0.0
                )

                if ok is True:
                    self._consecutive_failures = 0
                    status = "SUCCESS"
                    success = True
                    err = None
                elif ok is False:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= int(
                        cfg["execution_max_consecutive_failures"]
                    ):
                        self._disabled = True
                    status = "FAILED"
                    success = False
                    err = cerr
                else:
                    status = "UNKNOWN"
                    success = False
                    err = cerr

                self._last_exec_mono = time.monotonic()
                ms = int((time.perf_counter() - t_start) * 1000)
                res = ExecutionResult(
                    success=success,
                    status=status,
                    tx_signature=sig_str,
                    slippage_real=slip_real,
                    slippage_expected_bps=slippage_bps,
                    expected_out_raw=expected_out,
                    real_out_raw=real_delta,
                    price_impact_pct=price_impact,
                    error=err,
                    execution_time_ms=ms,
                )
                logger.info(
                    f"[LIVE {side_label}] status={status} sig={sig_str} "
                    f"token={persist_token_mint[:8]}… expected={expected_out} "
                    f"real_delta={real_delta} slip_real={slip_real:.4f} "
                    f"impact={price_impact:.4f} ms={ms}"
                )
                await self._persist(db, persist_token_mint, persist_amount_in, mode, res)
                return res
            finally:
                await rpc.close()

    async def _persist(
        self,
        db: Optional[Any],
        token_mint: str,
        lamports: int,
        mode: str,
        res: ExecutionResult,
    ) -> None:
        if db is None:
            return
        try:
            row = res.to_log_row(token_mint, lamports, mode)
            await db.insert_execution_log(row)
        except Exception as e:
            logger.error(f"No se pudo guardar execution_log: {e}")
