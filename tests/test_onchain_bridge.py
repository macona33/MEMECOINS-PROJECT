"""Tests del puente on-chain del bot (sin red)."""

import base64
import pytest
from unittest.mock import patch

from src.trading.onchain_bridge import (
    BotOnchainBridge,
    sell_error_indicates_frozen_token_account,
    sell_error_indicates_no_tokens_to_sell,
)


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


def test_sell_error_no_tokens_to_sell():
    assert sell_error_indicates_no_tokens_to_sell("saldo SPL 0 para este mint (nada que vender)") is True
    assert sell_error_indicates_no_tokens_to_sell("Nothing to sell") is True
    assert sell_error_indicates_no_tokens_to_sell("slippage exceeded") is False


def test_sell_error_frozen_detection():
    assert sell_error_indicates_frozen_token_account("Program log: Error: Account is frozen") is True
    assert sell_error_indicates_frozen_token_account("custom program error: 0x11") is True
    assert sell_error_indicates_frozen_token_account("slippage exceeded") is False
    assert sell_error_indicates_frozen_token_account(None) is False


def test_account_data_to_raw_bytes_solders_style():
    """solders.Account devuelve data como bytes (no lista base64)."""

    class FakeAccount:
        data = bytes(100)
        owner = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    raw = BotOnchainBridge._account_data_to_raw_bytes(FakeAccount())
    assert raw == bytes(100)


def test_account_data_to_raw_bytes_json_tuple():
    buf = bytes(82)
    b64 = base64.b64encode(buf).decode("ascii")

    class FakeAccount:
        data = [b64, "base64"]
        owner = "x"

    raw = BotOnchainBridge._account_data_to_raw_bytes(FakeAccount())
    assert raw == buf


def test_parse_mint_freeze_authority_flag():
    # Minimal Mint layout needs >= 82 bytes; freeze_authority_option lives at offset 46..50.
    b = BotOnchainBridge()
    buf = bytearray(82)
    assert b._parse_mint_has_freeze_authority(bytes(buf)) is False
    buf[46:50] = (1).to_bytes(4, "little")
    assert b._parse_mint_has_freeze_authority(bytes(buf)) is True
    assert b._parse_mint_has_freeze_authority(b"") is None
