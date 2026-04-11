"""Carga de configuración de ejecución (env + SETTINGS)."""

import os
from typing import Any, Dict

from dotenv import load_dotenv

from config.settings import SETTINGS

load_dotenv()


def _merged_execution_slippage_bps() -> int:
    base = int(os.getenv("EXECUTION_SLIPPAGE_BPS") or SETTINGS.get("execution_slippage_bps", 400))
    if not SETTINGS.get("sync_execution_slippage_with_paper", True):
        return base
    paper_bps = int(round(float(SETTINGS.get("slippage_pct", 0.02)) * 10000))
    return max(base, paper_bps)


def get_execution_config() -> Dict[str, Any]:
    """Merge SETTINGS con variables de entorno (sin secretos en retorno)."""
    mode = (os.getenv("TRADING_MODE") or SETTINGS.get("trading_mode") or "paper").strip().lower()
    if mode not in ("paper", "live"):
        mode = "paper"

    live_env = os.getenv("LIVE_TRADING_ENABLED")
    if live_env is None:
        live_enabled = bool(SETTINGS.get("live_trading_enabled", False))
    else:
        live_enabled = live_env.strip().lower() in ("1", "true", "yes", "on")

    base = (os.getenv("JUPITER_API_BASE") or SETTINGS.get("jupiter_api_base") or "").strip().rstrip("/")

    return {
        "trading_mode": "live" if mode == "live" else "paper",
        "live_trading_enabled": live_enabled,
        "rpc_url": os.getenv("SOLANA_RPC_URL", "").strip(),
        "jupiter_api_base": base,
        "jupiter_quote_timeout_s": float(
            os.getenv("JUPITER_QUOTE_TIMEOUT_S") or SETTINGS.get("jupiter_quote_timeout_s", 20.0)
        ),
        "jupiter_swap_timeout_s": float(
            os.getenv("JUPITER_SWAP_TIMEOUT_S") or SETTINGS.get("jupiter_swap_timeout_s", 30.0)
        ),
        "solana_rpc_timeout_s": float(
            os.getenv("SOLANA_RPC_TIMEOUT_S") or SETTINGS.get("solana_rpc_timeout_s", 30.0)
        ),
        "execution_slippage_bps": _merged_execution_slippage_bps(),
        "execution_max_price_impact": float(
            os.getenv("EXECUTION_MAX_PRICE_IMPACT")
            or SETTINGS.get("execution_max_price_impact", 0.05)
        ),
        "execution_trade_cooldown_s": float(
            os.getenv("EXECUTION_TRADE_COOLDOWN_S")
            or SETTINGS.get("execution_trade_cooldown_s", 7.0)
        ),
        "execution_rpc_max_retries": int(
            os.getenv("EXECUTION_RPC_MAX_RETRIES")
            or SETTINGS.get("execution_rpc_max_retries", 2)
        ),
        "execution_confirm_timeout_s": float(
            os.getenv("EXECUTION_CONFIRM_TIMEOUT_S")
            or SETTINGS.get("execution_confirm_timeout_s", 20.0)
        ),
        "execution_max_consecutive_failures": int(
            os.getenv("EXECUTION_MAX_CONSECUTIVE_FAILURES")
            or SETTINGS.get("execution_max_consecutive_failures", 2)
        ),
        "paper_slippage_pct": float(SETTINGS.get("slippage_pct", 0.02)),
    }
