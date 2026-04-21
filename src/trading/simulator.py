"""
Trade Simulator - Ejecutor de paper trades.
Simula entrada y salida de posiciones sin capital real.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from src.storage import DatabaseManager
from src.data_sources import DexScreenerClient
from src.models import EVSCalculator
from .kelly import KellyCalculator, PositionSize
from config.settings import SETTINGS

if TYPE_CHECKING:
    from src.trading.onchain_bridge import BotOnchainBridge


class TradeStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class SimulatedTrade:
    """Representa un trade simulado."""
    token_address: str
    symbol: str
    entry_price: float
    position_size_usd: float
    kelly_fraction: float
    stop_price: float
    take_profit_price: float
    
    entry_time: datetime = field(default_factory=datetime.now)
    status: TradeStatus = TradeStatus.OPEN
    
    current_price: float = 0.0
    current_mfe: float = 0.0
    current_mae: float = 0.0
    
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    label: Optional[str] = None  # v2.0: path-dependent (pump/rug/neutral/breakeven)

    evs_at_entry: float = 0.0
    p_rug_at_entry: float = 0.0
    p_pump_at_entry: float = 0.0
    trade_id: Optional[int] = None  # ID en tabla trades (para trade_features)
    onchain_entry_sig: Optional[str] = None
    onchain_exit_sig: Optional[str] = None
    onchain_sol_spent: Optional[float] = None
    onchain_entry_fee_lamports: Optional[int] = None
    onchain_exit_fee_lamports: Optional[int] = None

    @property
    def pnl_pct(self) -> float:
        if self.exit_price:
            return (self.exit_price - self.entry_price) / self.entry_price
        return (self.current_price - self.entry_price) / self.entry_price if self.current_price else 0
    
    @property
    def pnl_usd(self) -> float:
        return self.position_size_usd * self.pnl_pct
    
    @property
    def duration_minutes(self) -> float:
        end = self.exit_time or datetime.now()
        return (end - self.entry_time).total_seconds() / 60
    
    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_address": self.token_address,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size_usd": self.position_size_usd,
            "kelly_fraction": self.kelly_fraction,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status.value,
            "pnl_pct": self.pnl_pct,
            "pnl_usd": self.pnl_usd,
            "mfe": self.current_mfe,
            "mae": self.current_mae,
            "exit_reason": self.exit_reason,
            "label": self.label,
            "duration_minutes": self.duration_minutes,
            "evs_at_entry": self.evs_at_entry,
            "p_rug_at_entry": self.p_rug_at_entry,
            "p_pump_at_entry": self.p_pump_at_entry,
        }


class TradeSimulator:
    """
    Simulador de paper trading.
    Gestiona apertura, monitoreo y cierre de trades simulados.
    """
    
    def __init__(
        self,
        db: DatabaseManager,
        dex_client: Optional[DexScreenerClient] = None,
        evs_calculator: Optional[EVSCalculator] = None,
        kelly_calculator: Optional[KellyCalculator] = None,
        onchain_bridge: Optional["BotOnchainBridge"] = None,
    ):
        self.db = db
        self.dex_client = dex_client or DexScreenerClient()
        self.evs_calculator = evs_calculator or EVSCalculator()
        self.kelly_calculator = kelly_calculator or KellyCalculator()
        self._onchain = onchain_bridge

        self._active_trades: Dict[str, SimulatedTrade] = {}
        self._capital = SETTINGS["initial_capital"]
        self._allocated = 0.0
        
        self._callbacks: List[Callable] = []
    
    @property
    def available_capital(self) -> float:
        return self._capital - self._allocated
    
    @property
    def active_trade_count(self) -> int:
        return len(self._active_trades)

    def set_capital(self, capital: float) -> None:
        """Actualiza el capital (ej. desde risk_state al cargar)."""
        self._capital = max(0, capital)
    
    def on_trade_event(self, callback: Callable) -> None:
        """Registra callback para eventos de trade."""
        self._callbacks.append(callback)
    
    async def _notify(self, event: str, trade: SimulatedTrade) -> None:
        for callback in self._callbacks:
            try:
                result = callback(event, trade)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    async def load_active_trades(self) -> None:
        """Carga trades activos desde la base de datos."""
        active = await self.db.get_active_trades()
        
        for trade_data in active:
            entry_time_str = trade_data.get("entry_time")
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(str(entry_time_str).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    entry_time = datetime.now()
            else:
                entry_time = datetime.now()
            
            trade = SimulatedTrade(
                token_address=trade_data["token_address"],
                symbol=trade_data.get("symbol", ""),
                entry_price=trade_data["entry_price"],
                position_size_usd=trade_data["position_size_usd"],
                kelly_fraction=trade_data["kelly_fraction"],
                stop_price=trade_data["stop_price"],
                take_profit_price=trade_data["take_profit_price"],
                entry_time=entry_time,
                current_price=trade_data.get("current_price", trade_data["entry_price"]),
                current_mfe=trade_data.get("current_mfe", 0),
                current_mae=trade_data.get("current_mae", 0),
            )
            self._active_trades[trade.token_address] = trade
            self._allocated += trade.position_size_usd
        
        logger.info(f"Loaded {len(self._active_trades)} active trades")
    
    def can_open_trade(self, position_size: float) -> tuple:
        """Verifica si se puede abrir un nuevo trade."""
        if self.active_trade_count >= SETTINGS["max_concurrent_trades"]:
            return False, "Max concurrent trades reached"
        
        if position_size > self.available_capital:
            return False, f"Insufficient capital: need ${position_size:.2f}, have ${self.available_capital:.2f}"
        
        return True, None
    
    async def open_trade(
        self,
        token_address: str,
        token_info: Dict[str, Any],
        evs_result: Any,
        features: Dict[str, float],
    ) -> Optional[SimulatedTrade]:
        """
        Abre un nuevo trade simulado.
        """
        if token_address in self._active_trades:
            logger.warning(f"Trade already open for {token_address}")
            return None

        block_re = bool(SETTINGS.get("block_reentry_same_token_after_close", True))
        env_be = os.getenv("BLOCK_REENTRY_SAME_TOKEN_AFTER_CLOSE")
        if env_be is not None:
            block_re = env_be.strip().lower() in ("1", "true", "yes", "on")
        if block_re:
            closed_n = await self.db.count_closed_trades_for_token(token_address)
            if closed_n > 0:
                logger.warning(
                    "Re-entrada bloqueada: ya hay {} trade(s) cerrado(s) para mint {}…",
                    closed_n,
                    token_address[:12],
                )
                return None

        sizing_capital = self.available_capital
        live_sizing = False
        if self._onchain and self._onchain.is_active():
            live_sizing = True
            cap_usd, cap_err = await self._onchain.wallet_available_usd_for_new_buy(self.dex_client)
            if cap_usd is None:
                logger.warning("Sizing live: no se pudo leer capital wallet: {}", cap_err)
            else:
                sizing_capital = float(cap_usd)

        # En live, permite tamaños mínimos más pequeños (configurable) porque el cap en SOL ya protege.
        old_min = float(getattr(self.kelly_calculator, "min_position_usd", 50.0))
        old_max_pct = float(getattr(self.kelly_calculator, "max_position_pct", SETTINGS.get("max_position_pct", 0.03)))
        if live_sizing:
            self.kelly_calculator.min_position_usd = float(SETTINGS.get("min_position_usd_live", 10.0))
            self.kelly_calculator.max_position_pct = float(SETTINGS.get("max_position_pct_live", old_max_pct))
        else:
            self.kelly_calculator.min_position_usd = float(SETTINGS.get("min_position_usd_paper", old_min))
        try:
            position = self.kelly_calculator.calculate_position(
                evs_result,
                features,
                sizing_capital,
            )
        finally:
            self.kelly_calculator.min_position_usd = old_min
            self.kelly_calculator.max_position_pct = old_max_pct
        
        if not position.is_valid:
            logger.debug(f"Position not valid: {position.rationale}")
            return None
        
        can_open, reason = self.can_open_trade(position.position_usd)
        if not can_open:
            logger.warning(f"Cannot open trade: {reason}")
            return None
        
        entry_price = token_info.get("price_usd", 0)
        if entry_price <= 0:
            logger.error(f"Invalid entry price for {token_address}")
            return None
        
        stops = self.kelly_calculator.calculate_stop_levels(
            entry_price, evs_result, features
        )
        
        trade = SimulatedTrade(
            token_address=token_address,
            symbol=token_info.get("symbol", ""),
            entry_price=entry_price,
            position_size_usd=position.position_usd,
            kelly_fraction=position.kelly_fraction,
            stop_price=stops["stop_price"],
            take_profit_price=stops["take_profit_price"],
            current_price=entry_price,
            evs_at_entry=evs_result.evs_adj,
            p_rug_at_entry=evs_result.p_rug,
            p_pump_at_entry=evs_result.p_pump,
        )

        if self._onchain and self._onchain.is_active():
            buy_res = await self._onchain.execute_buy_for_open(
                token_address,
                position.position_usd,
                self.dex_client,
                self.db,
            )
            if not buy_res.ok:
                logger.error(
                    "On-chain BUY falló; trade no abierto (paper sin posición): {}",
                    buy_res.error,
                )
                return None
            trade.onchain_entry_sig = buy_res.tx_signature
            trade.onchain_sol_spent = buy_res.sol_amount
            trade.onchain_entry_fee_lamports = buy_res.fee_lamports

        self._active_trades[token_address] = trade
        self._allocated += position.position_usd

        trade_id = await self.db.open_trade({
            "token_address": token_address,
            "entry_price": entry_price,
            "position_size_usd": position.position_usd,
            "kelly_fraction": position.kelly_fraction,
            "stop_price": stops["stop_price"],
            "take_profit_price": stops["take_profit_price"],
            "evs_at_entry": evs_result.evs_adj,
            "p_rug_at_entry": evs_result.p_rug,
            "p_pump_at_entry": evs_result.p_pump,
        })
        trade.trade_id = trade_id
        
        logger.info(
            f"Opened trade: {trade.symbol} @ ${entry_price:.8f}, "
            f"Size: ${position.position_usd:.2f}, Kelly: {position.kelly_fraction:.2%}"
        )
        
        await self._notify("trade_opened", trade)
        
        return trade
    
    async def update_trade(
        self,
        token_address: str,
        current_price: float,
        volume: float = None,
        liquidity: float = None,
    ) -> Optional[str]:
        """
        Actualiza un trade con nuevo precio.
        Retorna razón de cierre si aplica.
        """
        trade = self._active_trades.get(token_address)
        if not trade or not trade.is_open:
            return None
        
        trade.current_price = current_price
        
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        
        if pnl_pct > trade.current_mfe:
            trade.current_mfe = pnl_pct
        if pnl_pct < trade.current_mae:
            trade.current_mae = pnl_pct
        
        close_reason = self._check_exit_conditions(trade, current_price)
        
        await self.db.update_active_trade(token_address, {
            "current_price": current_price,
            "current_mfe": trade.current_mfe,
            "current_mae": trade.current_mae,
        })
        
        if close_reason:
            await self.close_trade(token_address, current_price, close_reason)
        
        return close_reason
    
    def _check_exit_conditions(
        self, 
        trade: SimulatedTrade, 
        current_price: float
    ) -> Optional[str]:
        """Verifica condiciones de salida."""
        if current_price <= trade.stop_price:
            return "stop_loss"
        
        if current_price >= trade.take_profit_price:
            return "take_profit"
        
        if trade.duration_minutes > SETTINGS["max_hold_hours"] * 60:
            return "timeout"
        
        if trade.current_mfe > 0.15:
            trailing_stop = trade.entry_price * (1 + trade.current_mfe * 0.5)
            if current_price < trailing_stop:
                return "trailing_stop"
        
        return None
    
    async def close_trade(
        self,
        token_address: str,
        exit_price: float,
        reason: str,
    ) -> Optional[SimulatedTrade]:
        """
        Cierra un trade simulado.
        """
        trade = self._active_trades.get(token_address)
        if not trade:
            return None

        if self._onchain and self._onchain.is_active():
            sell_res = await self._onchain.execute_sell_for_close(token_address, self.db)
            if not sell_res.ok:
                logger.error(
                    "On-chain SELL al cerrar falló tras {} intentos (paper/db se cierran igual): {}",
                    sell_res.attempts,
                    sell_res.error,
                )
            elif sell_res.tx_signature:
                trade.onchain_exit_sig = sell_res.tx_signature
                trade.onchain_exit_fee_lamports = sell_res.fee_lamports

        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_reason = reason
        trade.status = TradeStatus.CLOSED

        from src.calibration.labels import LabelGenerator
        label_gen = LabelGenerator()
        label = label_gen.generate_label(
            mfe=trade.current_mfe,
            mae=trade.current_mae,
            pnl=trade.pnl_pct,
        )
        trade.label = label

        await self.db.close_trade(token_address, {
            "exit_price": exit_price,
            "pnl_pct": trade.pnl_pct,
            "pnl_usd": trade.pnl_usd,
            "mfe": trade.current_mfe,
            "mae": trade.current_mae,
            "label": label,
            "exit_reason": reason,
            "duration_minutes": trade.duration_minutes,
            "evs_at_entry": trade.evs_at_entry,
            "p_rug_at_entry": trade.p_rug_at_entry,
            "p_pump_at_entry": trade.p_pump_at_entry,
        })
        
        self._allocated -= trade.position_size_usd
        self._capital += trade.pnl_usd
        del self._active_trades[token_address]

        logger.info(
            f"Closed trade: {trade.symbol} @ ${exit_price:.8f}, "
            f"PnL: {trade.pnl_pct:.1%} (${trade.pnl_usd:.2f}), "
            f"Reason: {reason}, Label: {label}"
        )
        
        await self._notify("trade_closed", trade)
        
        return trade
    
    async def close_all_trades(self, reason: str = "manual_close") -> List[SimulatedTrade]:
        """Cierra todos los trades activos."""
        closed = []
        
        for address in list(self._active_trades.keys()):
            trade = self._active_trades[address]
            
            prices = await self.dex_client.get_prices_batch([address])
            current_price = prices.get(address, trade.current_price)
            
            closed_trade = await self.close_trade(address, current_price, reason)
            if closed_trade:
                closed.append(closed_trade)
        
        return closed
    
    def get_active_trades(self) -> List[SimulatedTrade]:
        """Retorna lista de trades activos."""
        return list(self._active_trades.values())
    
    def get_trade(self, token_address: str) -> Optional[SimulatedTrade]:
        """Obtiene un trade específico."""
        return self._active_trades.get(token_address)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Genera resumen del portfolio actual."""
        trades = self.get_active_trades()
        
        if not trades:
            return {
                "active_trades": 0,
                "total_allocated": 0,
                "total_pnl_usd": 0,
                "total_pnl_pct": 0,
                "available_capital": self._capital,
            }
        
        total_pnl = sum(t.pnl_usd for t in trades)
        total_allocated = sum(t.position_size_usd for t in trades)
        
        return {
            "active_trades": len(trades),
            "total_allocated": total_allocated,
            "total_pnl_usd": total_pnl,
            "total_pnl_pct": total_pnl / total_allocated if total_allocated > 0 else 0,
            "available_capital": self.available_capital,
            "best_trade": max(trades, key=lambda t: t.pnl_pct).symbol if trades else None,
            "worst_trade": min(trades, key=lambda t: t.pnl_pct).symbol if trades else None,
        }
    
    async def close(self) -> None:
        """Cierra el simulador."""
        await self.dex_client.close()
