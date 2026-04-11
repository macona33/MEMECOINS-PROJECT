"""Tests del puente on-chain del bot (sin red)."""

import pytest
from unittest.mock import patch

from src.trading.onchain_bridge import BotOnchainBridge


def _live_cfg():
    return {
        "trading_mode": "live",
        "live_trading_enabled": True,
        "rpc_url": "https://example.invalid",
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


@pytest.mark.asyncio
async def test_bridge_inactive_in_paper_mode():
    with patch("src.trading.onchain_bridge.get_execution_config", return_value={**_live_cfg(), "trading_mode": "paper"}):
        b = BotOnchainBridge()
        assert b.is_active() is False


@pytest.mark.asyncio
async def test_bridge_active_requires_flag():
    with patch.dict("os.environ", {"BOT_ONCHAIN_EXECUTION": "1"}, clear=False):
        with patch("src.trading.onchain_bridge.get_execution_config", return_value=_live_cfg()):
            b = BotOnchainBridge()
            assert b.is_active() is True


@pytest.mark.asyncio
async def test_compute_buy_sol_respects_cap():
    class FakeDex:
        async def get_prices_batch(self, addresses):
            return {addresses[0]: 100.0}

    b = BotOnchainBridge()
    with patch.object(b, "max_sol_per_trade", return_value=0.02):
        amt = await b.compute_buy_sol_amount(10_000.0, FakeDex())
        assert amt == pytest.approx(0.02)


def test_merged_slippage_bps_uses_paper_minimum():
    from src.execution import config as ec

    with patch.object(ec, "SETTINGS", {"execution_slippage_bps": 50, "slippage_pct": 0.02, "sync_execution_slippage_with_paper": True}):
        with patch.dict("os.environ", {"EXECUTION_SLIPPAGE_BPS": ""}, clear=False):
            assert ec._merged_execution_slippage_bps() == 200
    with patch.object(ec, "SETTINGS", {"execution_slippage_bps": 400, "slippage_pct": 0.02, "sync_execution_slippage_with_paper": True}):
        with patch.dict("os.environ", {"EXECUTION_SLIPPAGE_BPS": ""}, clear=False):
            assert ec._merged_execution_slippage_bps() == 400
