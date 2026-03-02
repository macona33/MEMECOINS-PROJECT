"""
Notificaciones externas (Discord, etc.) sin bloquear el flujo de trading.
"""

from .discord import notify_trade_opened, notify_trade_closed, send_daily_pnl_chart

__all__ = ["notify_trade_opened", "notify_trade_closed", "send_daily_pnl_chart"]
