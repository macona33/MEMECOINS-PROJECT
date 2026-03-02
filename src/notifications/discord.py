"""
Notificaciones a Discord vía webhook. Ejecución en segundo plano para no afectar latencia.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
            {"name": "Address", "value": f"`{trade.token_address}`", "inline": False},
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
            {"name": "Address", "value": f"`{trade.token_address}`", "inline": False},
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


def _generate_pnl_chart(equity_curve: List[Dict[str, Any]], initial_capital: float) -> Path:
    """Genera gráfico de PnL y retorna ruta al archivo PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if not equity_curve:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Sin datos de trades aún", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    else:
        dates = []
        equities = []
        for p in equity_curve:
            if p.get("date"):
                try:
                    dates.append(datetime.strptime(p["date"], "%Y-%m-%d"))
                except (ValueError, TypeError):
                    continue
            else:
                dates.append(datetime.now())
            equities.append(p.get("equity", initial_capital))

        if not dates:
            dates = [datetime.now()]
            equities = [initial_capital]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, equities, color="#00d26a", linewidth=2, marker="o", markersize=4)
        ax.axhline(y=initial_capital, color="#666", linestyle="--", alpha=0.7, label="Capital inicial")
        ax.set_ylabel("Equity (USD)", fontsize=11)
        ax.set_xlabel("Fecha", fontsize=11)
        ax.set_title("Paper Trading - Curva de Equity", fontsize=13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()

    path = Path(tempfile.gettempdir()) / f"pnl_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


async def send_daily_pnl_chart(
    equity_curve: List[Dict[str, Any]],
    initial_capital: float = 10000,
) -> None:
    """
    Genera gráfico de PnL y lo envía a Discord.
    Llamar desde daily_report o scheduler diario.
    """
    if not DISCORD_WEBHOOK_URL:
        logger.debug("Discord webhook not configured, skipping PnL chart")
        return
    try:
        path = _generate_pnl_chart(equity_curve, initial_capital)
        import aiohttp
        last_equity = equity_curve[-1].get("equity", initial_capital) if equity_curve else initial_capital
        embed = {
            "title": "📊 Reporte Diario - Paper Trading",
            "description": f"Equity actual: **${last_equity:,.2f}** (inicial: ${initial_capital:,.2f})",
            "color": 0x00D26A,
            "timestamp": datetime.now().isoformat(),
            "image": {"url": "attachment://pnl_chart.png"},
        }
        payload = {"embeds": [embed]}
        timeout = aiohttp.ClientTimeout(total=15)
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("payload_json", json.dumps(payload))
            data.add_field("file", f, filename="pnl_chart.png", content_type="image/png")
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(DISCORD_WEBHOOK_URL, data=data) as resp:
                    if resp.status >= 400:
                        logger.warning(f"Discord chart upload status {resp.status}: {await resp.text()}")
        path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to send PnL chart to Discord: {e}")
