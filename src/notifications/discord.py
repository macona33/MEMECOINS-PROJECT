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


def _parse_chart_event_time(raw: Optional[str]):
    """Parsea marca temporal para eje X (naive local)."""
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip().replace("T", " ")[:19]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _generate_pnl_chart(
    equity_curve: List[Dict[str, Any]],
    initial_capital: float,
    *,
    y_axis_label: str = "PnL ($)",
    title: str = "Rendimiento del bot",
    cumulative_from_zero: bool = False,
) -> Path:
    """
    Curva de PnL: eje Y = PnL acumulado respecto a initial_capital, o valor acumulado si cumulative_from_zero.
    Eje X = tiempo (event_time) si todas las filas lo traen; si no, índice de evento.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if not equity_curve:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.text(0.5, 0.5, "Sin datos en el periodo", ha="center", va="center", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    else:
        if cumulative_from_zero or abs(initial_capital) < 1e-9:
            pnl_usd = [float(p.get("equity") or 0) for p in equity_curve]
        else:
            pnl_usd = [
                float(p.get("equity", initial_capital) or 0) - float(initial_capital)
                for p in equity_curve
            ]

        xs_dt = [_parse_chart_event_time(p.get("event_time")) for p in equity_curve]
        use_time = all(x is not None for x in xs_dt) and len(xs_dt) > 1

        fig, ax = plt.subplots(figsize=(10, 5.2))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4", labelsize=9)
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#89b4fa")
        ax.spines["bottom"].set_color("#45475a")
        ax.spines["left"].set_color("#45475a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if use_time:
            x_num = [mdates.date2num(d) for d in xs_dt]
            ax.plot(
                x_num,
                pnl_usd,
                color="#94e2d5",
                linewidth=2.2,
                marker="o",
                markersize=4,
                label="PnL acumulado",
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate(rotation=22)
            ax.set_xlabel("Tiempo (cierre / ejecución)", fontsize=10, color="#cdd6f4")
        else:
            trades_t = list(range(len(equity_curve)))
            ax.plot(
                trades_t,
                pnl_usd,
                color="#94e2d5",
                linewidth=2.2,
                marker="o",
                markersize=4,
                label="PnL acumulado",
            )
            ax.set_xlabel("Evento nº (orden cronológico)", fontsize=10, color="#cdd6f4")

        ax.axhline(y=0, color="#6c7086", linestyle="--", linewidth=1)
        ax.set_ylabel(y_axis_label, fontsize=10, color="#cdd6f4")
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(True, alpha=0.25, color="#45475a")
        ax.legend(loc="upper left", framealpha=0.85, facecolor="#313244", edgecolor="#45475a")
        if pnl_usd:
            y_min = min(min(pnl_usd), 0)
            y_max = max(max(pnl_usd), 0)
            margin = (y_max - y_min) * 0.08 if (y_max - y_min) > 1e-9 else max(abs(y_max), 1.0) * 0.05
            ax.set_ylim(bottom=y_min - margin, top=y_max + margin)
        fig.tight_layout()

    path = Path(tempfile.gettempdir()) / f"pnl_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


async def send_daily_pnl_chart(
    equity_curve: List[Dict[str, Any]],
    initial_capital: float = 10000,
    *,
    report_mode: str = "paper",
    chart_ylabel: Optional[str] = None,
    chart_title: Optional[str] = None,
    cumulative_from_zero: bool = False,
    baseline_note: Optional[str] = None,
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
            "PnL acumulado aprox. (USD, SOL×precio ref.)"
            if report_mode == "live"
            else "PnL acumulado (USD, trades cerrados)"
        )
        ttl = chart_title or (
            "Reporte diario — Live (flujo swaps)" if report_mode == "live" else "Reporte diario — Paper (DB)"
        )
        path = _generate_pnl_chart(
            equity_curve,
            initial_capital,
            y_axis_label=ylabel,
            title=ttl,
            cumulative_from_zero=cumulative_from_zero,
        )
        import aiohttp
        last_y = float(equity_curve[-1].get("equity", 0) or 0) if equity_curve else 0.0
        if report_mode == "live":
            title = "📊 " + ttl
            desc = (
                "Flujo **neto aproximado** en USD desde swaps registrados (compras −SOL / ventas +SOL × precio). "
                "**No** es el saldo total de la cartera ni incluye SOL que ya tenías.\n"
                f"**Acumulado (serie):** `${last_y:,.2f}`"
            )
        else:
            title = "📊 " + ttl
            if cumulative_from_zero:
                desc = (
                    "PnL **acumulado en USD** desde la línea base (solo trades cerrados posteriores).\n"
                    f"**Total desde reset:** `${last_y:,.2f}`"
                )
            else:
                desc = (
                    "Equity del **modelo** (capital inicial + suma de `pnl_usd` por cierre). "
                    "Coincide con Kelly/risk_state si el simulador y la DB están alineados.\n"
                    f"**Equity final (serie):** `${last_y:,.2f}` · inicial ref.: `${initial_capital:,.2f}`"
                )
        if baseline_note:
            desc = desc + "\n" + baseline_note
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
