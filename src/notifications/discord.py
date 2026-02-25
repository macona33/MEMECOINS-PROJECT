"""
Notificaciones a Discord vía webhook. Ejecución en segundo plano para no afectar latencia.
"""

import asyncio
import os
from typing import Any, Dict

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
# Webhook desde variable de entorno (pon la URL en tu archivo .env)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


def _embed_trade_opened(trade: Any) -> Dict[str, Any]:
    """Construye el embed de Discord para trade abierto."""
    return {
        "title": "Trade abierto",
        "color": 0x00FF00,  # Verde
        "fields": [
            {"name": "Token", "value": trade.symbol, "inline": True},
            {"name": "Precio entrada", "value": f"${trade.entry_price:.8f}", "inline": True},
            {"name": "Tamaño", "value": f"${trade.position_size_usd:.2f}", "inline": True},
            {"name": "Kelly", "value": f"{trade.kelly_fraction:.2%}", "inline": True},
            {"name": "Stop", "value": f"${trade.stop_price:.8f}", "inline": True},
            {"name": "Take profit", "value": f"${trade.take_profit_price:.8f}", "inline": True},
            {"name": "EVS entrada", "value": f"{trade.evs_at_entry:.4f}", "inline": True},
            {"name": "P(rug)", "value": f"{trade.p_rug_at_entry:.2%}", "inline": True},
            {"name": "P(pump)", "value": f"{trade.p_pump_at_entry:.2%}", "inline": True},
        ],
        "timestamp": trade.entry_time.isoformat(),
    }


def _embed_trade_closed(trade: Any) -> Dict[str, Any]:
    """Construye el embed de Discord para trade cerrado."""
    is_win = trade.pnl_usd > 0
    color = 0x00FF00 if is_win else 0xFF0000  # Verde / Rojo
    return {
        "title": "Trade cerrado",
        "color": color,
        "fields": [
            {"name": "Token", "value": trade.symbol, "inline": True},
            {"name": "Motivo", "value": trade.exit_reason or "—", "inline": True},
            {"name": "PnL %", "value": f"{trade.pnl_pct:.2%}", "inline": True},
            {"name": "PnL USD", "value": f"${trade.pnl_usd:.2f}", "inline": True},
            {"name": "Precio entrada", "value": f"${trade.entry_price:.8f}", "inline": True},
            {"name": "Precio salida", "value": f"${trade.exit_price:.8f}" if trade.exit_price else "—", "inline": True},
            {"name": "Tamaño", "value": f"${trade.position_size_usd:.2f}", "inline": True},
            {"name": "Duración", "value": f"{trade.duration_minutes:.1f} min", "inline": True},
            {"name": "MFE", "value": f"{trade.current_mfe:.2%}", "inline": True},
            {"name": "MAE", "value": f"{trade.current_mae:.2%}", "inline": True},
            {"name": "EVS entrada", "value": f"{trade.evs_at_entry:.4f}", "inline": True},
        ],
        "timestamp": (trade.exit_time.isoformat() if trade.exit_time else None),
    }


async def _send_webhook(embed: Dict[str, Any]) -> None:
    """Envía un mensaje al webhook de Discord. No lanza excepciones."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        import aiohttp
        payload = {"embeds": [embed]}
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(DISCORD_WEBHOOK_URL, json=payload) as resp:
                if resp.status >= 400:
                    logger.warning(f"Discord webhook status {resp.status}: {await resp.text()}")
    except asyncio.TimeoutError:
        logger.debug("Discord webhook timeout (ignored)")
    except Exception as e:
        logger.debug(f"Discord notification failed: {e}")


def notify_trade_opened(trade: Any) -> None:
    """
    Programa el envío de notificación de trade abierto a Discord.
    No bloquea: se ejecuta en segundo plano y no afecta la latencia del trading.
    """
    if not DISCORD_WEBHOOK_URL:
        return
    embed = _embed_trade_opened(trade)
    asyncio.create_task(_send_webhook(embed))


def notify_trade_closed(trade: Any) -> None:
    """
    Programa el envío de notificación de trade cerrado a Discord.
    No bloquea: se ejecuta en segundo plano.
    """
    if not DISCORD_WEBHOOK_URL:
        return
    embed = _embed_trade_closed(trade)
    asyncio.create_task(_send_webhook(embed))
