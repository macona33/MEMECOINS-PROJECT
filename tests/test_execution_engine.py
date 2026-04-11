"""Tests del ExecutionEngine_v1_Jupiter (sin red)."""

import pytest
from unittest.mock import AsyncMock, patch

from src.execution.engine import ExecutionEngineV1Jupiter
from src.execution.jupiter_client import JupiterV6Client


def _base_cfg(**over):
    base = {
        "trading_mode": "paper",
        "live_trading_enabled": False,
        "rpc_url": "",
        "jupiter_api_base": "https://lite-api.jup.ag/swap/v1",
        "jupiter_quote_timeout_s": 5.0,
        "jupiter_swap_timeout_s": 5.0,
        "solana_rpc_timeout_s": 5.0,
        "execution_slippage_bps": 400,
        "execution_max_price_impact": 0.05,
        "execution_trade_cooldown_s": 0.0,
        "execution_rpc_max_retries": 2,
        "execution_confirm_timeout_s": 5.0,
        "execution_max_consecutive_failures": 2,
        "paper_slippage_pct": 0.02,
    }
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_paper_sim_success():
    quote = {
        "routePlan": [{"swaps": []}],
        "outAmount": "1000000",
        "priceImpactPct": "0.01",
    }
    with patch("src.execution.engine.get_execution_config", return_value=_base_cfg()):
        with patch.object(JupiterV6Client, "get_quote", new=AsyncMock(return_value=quote)):
            eng = ExecutionEngineV1Jupiter()
            res = await eng.execute_trade("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 0.01, db=None)
    assert res.success is True
    assert res.status == "PAPER_SIM"
    assert res.tx_signature == "SIMULATED"
    assert res.expected_out_raw == 1_000_000
    assert res.real_out_raw == 980_000
    assert res.slippage_real is not None
    assert abs(res.slippage_real - 0.02) < 1e-9


@pytest.mark.asyncio
async def test_paper_sell_sim_success():
    quote = {
        "routePlan": [{"swaps": []}],
        "outAmount": "5000000",
        "priceImpactPct": "0.02",
    }
    raw_in = 1_000_000_000
    with patch("src.execution.engine.get_execution_config", return_value=_base_cfg()):
        with patch.object(JupiterV6Client, "get_quote", new=AsyncMock(return_value=quote)):
            eng = ExecutionEngineV1Jupiter()
            res = await eng.execute_sell("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", raw_in, db=None)
    assert res.success is True
    assert res.status == "PAPER_SIM"
    assert res.expected_out_raw == 5_000_000
    assert res.real_out_raw == 4_900_000


@pytest.mark.asyncio
async def test_execute_sell_all_rejects_zero_balance():
    with patch("src.execution.engine.get_execution_config", return_value=_base_cfg()):
        eng = ExecutionEngineV1Jupiter()
        with patch.object(
            eng,
            "get_wallet_token_balance_raw",
            new=AsyncMock(return_value=(0, None)),
        ):
            res = await eng.execute_sell_all("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", db=None)
    assert res.success is False
    assert res.status == "REJECTED"
    assert "nada que vender" in (res.error or "")


@pytest.mark.asyncio
async def test_execute_sell_all_calls_execute_sell():
    with patch("src.execution.engine.get_execution_config", return_value=_base_cfg()):
        eng = ExecutionEngineV1Jupiter()
        with patch.object(
            eng,
            "get_wallet_token_balance_raw",
            new=AsyncMock(return_value=(12345, None)),
        ):
            with patch.object(eng, "execute_sell", new=AsyncMock()) as m:
                await eng.execute_sell_all("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", db=None)
    m.assert_called_once_with("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 12345, db=None)


@pytest.mark.asyncio
async def test_reject_price_impact():
    quote = {
        "routePlan": [{"swaps": []}],
        "outAmount": "1000000",
        "priceImpactPct": "0.10",
    }
    with patch("src.execution.engine.get_execution_config", return_value=_base_cfg()):
        with patch.object(JupiterV6Client, "get_quote", new=AsyncMock(return_value=quote)):
            eng = ExecutionEngineV1Jupiter()
            res = await eng.execute_trade("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 0.01, db=None)
    assert res.success is False
    assert res.status == "REJECTED"
    assert "priceImpactPct" in (res.error or "")


@pytest.mark.asyncio
async def test_sign_jupiter_versioned_smoke():
    from solders.keypair import Keypair
    from solders.message import MessageV0
    from solders.hash import Hash
    from solders.instruction import Instruction
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction
    import base64

    from src.execution.solana_tx import sign_jupiter_versioned_transaction

    kp = Keypair()
    m = MessageV0.try_compile(
        kp.pubkey(),
        [Instruction(Pubkey.default(), b"", [])],
        [],
        Hash.new_unique(),
    )
    tx = VersionedTransaction(m, [kp])
    b64 = base64.b64encode(bytes(tx)).decode("ascii")
    signed = sign_jupiter_versioned_transaction(b64, kp)
    assert signed.verify_with_results() is not None
