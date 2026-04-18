"""
Notificaciones a Discord vía webhook. Ejecución en segundo plano para no afectar latencia.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
# Webhook desde variable de entorno (pon la URL en tu archivo .env)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


def _solscan_tx_url(sig: Optional[str]) -> Optional[str]:
    if not sig or sig == "SIMULATED":
        return None
    return f"https://solscan.io/tx/{sig}"


def _embed_trade_opened(trade: Any) -> Dict[str, Any]:
    """Construye el embed de Discord para trade abierto."""
    fields: List[Dict[str, Any]] = [
        {"name": "Token", "value": trade.symbol, "inline": True},
        {"name": "Address", "value": f"`{trade.token_address}`", "inline": False},
        {"name": "Precio entrada (Dex)", "value": f"${trade.entry_price:.8f}", "inline": True},
        {
            "name": "Tamaño modelo (Kelly / equity DB)",
            "value": f"${trade.position_size_usd:.2f} — *no es el saldo de tu wallet*",
            "inline": True,
        },
        {"name": "Kelly", "value": f"{trade.kelly_fraction:.2%}", "inline": True},
        {"name": "Stop", "value": f"${trade.stop_price:.8f}", "inline": True},
        {"name": "Take profit", "value": f"${trade.take_profit_price:.8f}", "inline": True},
        {"name": "EVS entrada", "value": f"{trade.evs_at_entry:.4f}", "inline": True},
        {"name": "P(rug)", "value": f"{trade.p_rug_at_entry:.2%}", "inline": True},
        {"name": "P(pump)", "value": f"{trade.p_pump_at_entry:.2%}", "inline": True},
    ]
    sig = getattr(trade, "onchain_entry_sig", None)
    if sig and sig != "SIMULATED":
        sol_spent = getattr(trade, "onchain_sol_spent", None)
        fee = getattr(trade, "onchain_entry_fee_lamports", None)
        parts = []
        if sol_spent is not None:
            parts.append(f"**SOL gastados en swap:** `{sol_spent:.6f}`")
        if fee is not None:
            parts.append(f"**Fee tx (entrada):** `{fee}` lamports (~{fee / 1e9:.6f} SOL)")
        url = _solscan_tx_url(sig)
        if url:
            parts.append(f"**Explorer:** {url}")
        fields.append(
            {
                "name": "Wallet live (Solana)",
                "value": "\n".join(parts) if parts else f"`{sig}`",
                "inline": False,
            }
        )
    else:
        fields.append(
            {
                "name": "Wallet live",
                "value": "Sin compra on-chain en este evento (paper / flags off).",
                "inline": False,
            }
        )
    return {
        "title": "Trade abierto",
        "description": "Señales y tamaño **modelo**; bloque *Wallet live* = ejecución real si aplica.",
        "color": 0x00FF00,
        "fields": fields,
        "timestamp": trade.entry_time.isoformat(),
    }


def _embed_trade_closed(trade: Any) -> Dict[str, Any]:
    """Construye el embed de Discord para trade cerrado."""
    is_win = trade.pnl_usd > 0
    color = 0x00FF00 if is_win else 0xFF0000  # Verde / Rojo
    fields: List[Dict[str, Any]] = [
        {"name": "Token", "value": trade.symbol, "inline": True},
        {"name": "Address", "value": f"`{trade.token_address}`", "inline": False},
        {"name": "Motivo", "value": trade.exit_reason or "—", "inline": True},
        {"name": "PnL % (modelo Dex)", "value": f"{trade.pnl_pct:.2%}", "inline": True},
        {
            "name": "PnL USD (modelo / equity DB)",
            "value": f"${trade.pnl_usd:.2f} — *no es ganancia en tu wallet*",
            "inline": True,
        },
        {"name": "Precio entrada", "value": f"${trade.entry_price:.8f}", "inline": True},
        {"name": "Precio salida", "value": f"${trade.exit_price:.8f}" if trade.exit_price else "—", "inline": True},
        {
            "name": "Tamaño modelo",
            "value": f"${trade.position_size_usd:.2f}",
            "inline": True,
        },
        {"name": "Duración", "value": f"{trade.duration_minutes:.1f} min", "inline": True},
        {"name": "MFE", "value": f"{trade.current_mfe:.2%}", "inline": True},
        {"name": "MAE", "value": f"{trade.current_mae:.2%}", "inline": True},
        {"name": "EVS entrada", "value": f"{trade.evs_at_entry:.4f}", "inline": True},
    ]
    xs = getattr(trade, "onchain_exit_sig", None)
    if xs and xs != "SIMULATED":
        fee_out = getattr(trade, "onchain_exit_fee_lamports", None)
        parts = []
        if fee_out is not None:
            parts.append(f"**Fee tx (salida):** `{fee_out}` lamports (~{fee_out / 1e9:.6f} SOL)")
        url = _solscan_tx_url(xs)
        if url:
            parts.append(f"**Explorer salida:** {url}")
        xe = getattr(trade, "onchain_entry_sig", None)
        fee_in = getattr(trade, "onchain_entry_fee_lamports", None)
        if xe and fee_in is not None:
            parts.insert(0, f"**Fee tx (entrada):** `{fee_in}` lamports (~{fee_in / 1e9:.6f} SOL)")
        fields.append(
            {
                "name": "Wallet live (Solana)",
                "value": "\n".join(parts) if parts else f"`{xs}`",
                "inline": False,
            }
        )
    else:
        fields.append(
            {
                "name": "Wallet live",
                "value": "Sin venta on-chain registrada en este cierre (paper o venta fallida).",
                "inline": False,
            }
        )
    return {
        "title": "Trade cerrado",
        "description": "PnL y precios según **modelo** (Dex); fees = txs reales si constan.",
        "color": color,
        "fields": fields,
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


def _generate_pnl_chart(
    equity_curve: List[Dict[str, Any]],
    initial_capital: float,
    *,
    y_axis_label: str = "PnL ($)",
) -> Path:
    """Genera gráfico de PnL (estilo Strategy: Trades vs PnL $) y retorna ruta al archivo PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not equity_curve:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Sin datos de trades aún", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    else:
        trades_t = list(range(len(equity_curve)))
        pnl_usd = [(p.get("equity", initial_capital) - initial_capital) for p in equity_curve]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#f5f5f5")
        ax.set_facecolor("#f0f0f0")
        ax.tick_params(colors="#333")
        ax.xaxis.label.set_color("#333")
        ax.yaxis.label.set_color("#333")
        ax.title.set_color("#2d2d2d")

        ax.plot(
            trades_t,
            pnl_usd,
            color="#2dd4bf",
            linewidth=2,
            marker="o",
            markersize=5,
            label="portfolio pnl",
        )
        ax.axhline(y=0, color="#555", linestyle="--", linewidth=1)
        ax.set_ylabel(y_axis_label, fontsize=11)
        ax.set_xlabel("Trades (T)", fontsize=11)
        ax.set_title("Strategy", fontsize=13)
        ax.grid(True, alpha=0.4, color="#999")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_xlim(left=-0.5)
        if pnl_usd:
            y_min = min(min(pnl_usd), 0)
            y_max = max(max(pnl_usd), 0)
            margin = (y_max - y_min) * 0.05 or 10
            ax.set_ylim(bottom=y_min - margin, top=y_max + margin)
        fig.tight_layout()

    path = Path(tempfile.gettempdir()) / f"pnl_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


async def send_daily_pnl_chart(
    equity_curve: List[Dict[str, Any]],
    initial_capital: float = 10000,
    *,
    report_mode: str = "paper",
    chart_ylabel: Optional[str] = None,
) -> None:
    """
    Genera gráfico de PnL y lo envía a Discord.
    Llamar desde daily_report o scheduler diario.
    """
    if not DISCORD_WEBHOOK_URL:
        logger.debug("Discord webhook not configured, skipping PnL chart")
        return
    try:
        ylabel = chart_ylabel or (
            "PnL aprox. (USD, SOL×ref.)" if report_mode == "live" else "PnL ($)"
        )
        path = _generate_pnl_chart(equity_curve, initial_capital, y_axis_label=ylabel)
        import aiohttp
        last_equity = equity_curve[-1].get("equity", initial_capital) if equity_curve else initial_capital
        if report_mode == "live":
            title = "📊 Reporte Diario – Live (wallet / swaps)"
            desc = (
                f"Flujo neto acumulado aprox. en USD (no es el saldo total de la wallet): "
                f"**${last_equity:,.2f}** · referencia base en el gráfico: **${initial_capital:,.2f}**"
            )
        else:
            title = "📊 Reporte Diario – Paper (modelo / equity DB)"
            desc = (
                f"Equity modelo: **${last_equity:,.2f}** (inicial paper: **${initial_capital:,.2f}**)"
            )
        embed = {
            "title": title,
            "description": desc,
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
