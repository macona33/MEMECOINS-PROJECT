"""
Risk Manager v2.0.
Control dinámico de drawdown y exposición.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from config.settings import SETTINGS
from src.storage import DatabaseManager


@dataclass
class RiskState:
    """Estado actual de riesgo."""
    current_drawdown: float
    peak_equity: float
    current_equity: float
    consecutive_losses: int
    frozen_until: Optional[datetime]
    gamma_multiplier: float
    
    @property
    def is_frozen(self) -> bool:
        """Determina si el trading está congelado."""
        if self.frozen_until is None:
            return False
        return datetime.now() < self.frozen_until
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "consecutive_losses": self.consecutive_losses,
            "frozen_until": self.frozen_until.isoformat() if self.frozen_until else None,
            "gamma_multiplier": self.gamma_multiplier,
        }


class RiskManager:
    """
    Risk Manager v2.0 para control dinámico de drawdown.
    
    Reglas:
    1. Drawdown 10-20%: gamma *= 0.75
    2. Drawdown 20-30%: gamma *= 0.50
    3. Drawdown > 30%: Freeze por 2 horas
    4. 3 pérdidas consecutivas: Siguiente trade con 50% tamaño
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self._db = db
        self._state: Optional[RiskState] = None
        
        self._drawdown_level_1 = SETTINGS.get("drawdown_level_1", 0.10)
        self._drawdown_gamma_mult_1 = SETTINGS.get("drawdown_gamma_mult_1", 0.75)
        self._drawdown_level_2 = SETTINGS.get("drawdown_level_2", 0.20)
        self._drawdown_gamma_mult_2 = SETTINGS.get("drawdown_gamma_mult_2", 0.50)
        self._drawdown_freeze_level = SETTINGS.get("drawdown_freeze_level", 0.30)
        self._drawdown_freeze_hours = SETTINGS.get("drawdown_freeze_hours", 2)
        self._consecutive_loss_threshold = SETTINGS.get("consecutive_loss_threshold", 3)
        self._consecutive_loss_size_mult = SETTINGS.get("consecutive_loss_size_mult", 0.50)
        
        self._initial_capital = SETTINGS.get("initial_capital", 10000)
    
    def set_database(self, db: DatabaseManager) -> None:
        """Configura conexión a base de datos."""
        self._db = db
    
    @property
    def state(self) -> RiskState:
        """Estado actual de riesgo."""
        if self._state is None:
            self._state = RiskState(
                current_drawdown=0,
                peak_equity=self._initial_capital,
                current_equity=self._initial_capital,
                consecutive_losses=0,
                frozen_until=None,
                gamma_multiplier=1.0,
            )
        return self._state
    
    async def load_state(self) -> None:
        """Carga estado de riesgo desde la base de datos."""
        if not self._db:
            return
        
        db_state = await self._db.get_risk_state()
        
        frozen_until = None
        if db_state.get("frozen_until"):
            try:
                frozen_until = datetime.fromisoformat(db_state["frozen_until"])
            except (ValueError, TypeError):
                pass
        
        self._state = RiskState(
            current_drawdown=db_state.get("current_drawdown", 0),
            peak_equity=db_state.get("peak_equity", self._initial_capital),
            current_equity=db_state.get("current_equity", self._initial_capital),
            consecutive_losses=db_state.get("consecutive_losses", 0),
            frozen_until=frozen_until,
            gamma_multiplier=db_state.get("gamma_multiplier", 1.0),
        )
        
        logger.info(
            f"Loaded risk state: drawdown={self._state.current_drawdown:.1%}, "
            f"gamma_mult={self._state.gamma_multiplier:.2f}, "
            f"consecutive_losses={self._state.consecutive_losses}"
        )
    
    async def save_state(self) -> None:
        """Guarda estado de riesgo en la base de datos."""
        if not self._db or not self._state:
            return
        
        await self._db.update_risk_state({
            "current_drawdown": self._state.current_drawdown,
            "peak_equity": self._state.peak_equity,
            "current_equity": self._state.current_equity,
            "consecutive_losses": self._state.consecutive_losses,
            "frozen_until": self._state.frozen_until.isoformat() if self._state.frozen_until else None,
            "gamma_multiplier": self._state.gamma_multiplier,
        })
    
    def update_equity(self, new_equity: float) -> None:
        """
        Actualiza el equity y recalcula drawdown.
        
        Args:
            new_equity: Nuevo valor del equity
        """
        state = self.state
        state.current_equity = new_equity
        
        if new_equity > state.peak_equity:
            state.peak_equity = new_equity
            state.current_drawdown = 0
            logger.info(f"New equity peak: ${new_equity:.2f}")
        else:
            state.current_drawdown = (state.peak_equity - new_equity) / state.peak_equity
        
        self._recalculate_gamma()
        self._check_freeze()
    
    def record_trade_result(self, pnl_usd: float, is_win: bool) -> None:
        """
        Registra resultado de un trade.
        
        Args:
            pnl_usd: PnL en USD
            is_win: Si el trade fue ganador
        """
        state = self.state
        
        if is_win:
            state.consecutive_losses = 0
        else:
            state.consecutive_losses += 1
            logger.info(f"Consecutive losses: {state.consecutive_losses}")
        
        new_equity = state.current_equity + pnl_usd
        self.update_equity(new_equity)
    
    def _recalculate_gamma(self) -> None:
        """Recalcula el multiplicador de gamma basado en drawdown."""
        state = self.state
        dd = state.current_drawdown
        
        if dd >= self._drawdown_level_2:
            state.gamma_multiplier = self._drawdown_gamma_mult_2
        elif dd >= self._drawdown_level_1:
            state.gamma_multiplier = self._drawdown_gamma_mult_1
        else:
            state.gamma_multiplier = 1.0
        
        logger.debug(f"Gamma multiplier updated: {state.gamma_multiplier:.2f} (drawdown={dd:.1%})")
    
    def _check_freeze(self) -> None:
        """Verifica si se debe congelar el trading."""
        state = self.state
        
        if state.current_drawdown >= self._drawdown_freeze_level:
            if not state.is_frozen:
                state.frozen_until = datetime.now() + timedelta(hours=self._drawdown_freeze_hours)
                logger.warning(
                    f"Trading FROZEN until {state.frozen_until.isoformat()} "
                    f"(drawdown={state.current_drawdown:.1%} >= {self._drawdown_freeze_level:.1%})"
                )
    
    def can_trade(self) -> tuple:
        """
        Verifica si se puede abrir un nuevo trade.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        state = self.state
        
        if state.is_frozen:
            remaining = state.frozen_until - datetime.now()
            minutes_left = remaining.total_seconds() / 60
            return False, f"Trading frozen for {minutes_left:.0f} more minutes"
        
        return True, "OK"
    
    def get_position_multiplier(self) -> float:
        """
        Obtiene multiplicador de tamaño de posición.
        
        Considera:
        - Gamma multiplier por drawdown
        - Reducción por pérdidas consecutivas
        
        Returns:
            Multiplicador total para position size
        """
        state = self.state
        
        base_mult = state.gamma_multiplier
        
        if state.consecutive_losses >= self._consecutive_loss_threshold:
            base_mult *= self._consecutive_loss_size_mult
            logger.info(
                f"Position size reduced by {self._consecutive_loss_size_mult:.0%} "
                f"({state.consecutive_losses} consecutive losses)"
            )
        
        return base_mult
    
    def get_gamma_multiplier(self) -> float:
        """Obtiene multiplicador de gamma para Kelly calculator."""
        return self.state.gamma_multiplier * self.get_position_multiplier()
    
    def reset_freeze(self) -> None:
        """Resetea el freeze manualmente (para testing/admin)."""
        if self._state:
            self._state.frozen_until = None
            logger.info("Trading freeze manually reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del risk manager."""
        state = self.state
        can_trade, reason = self.can_trade()
        
        return {
            "current_equity": state.current_equity,
            "peak_equity": state.peak_equity,
            "current_drawdown": f"{state.current_drawdown:.1%}",
            "consecutive_losses": state.consecutive_losses,
            "gamma_multiplier": state.gamma_multiplier,
            "position_multiplier": self.get_position_multiplier(),
            "effective_gamma": self.get_gamma_multiplier(),
            "can_trade": can_trade,
            "trade_status": reason,
            "is_frozen": state.is_frozen,
            "frozen_until": state.frozen_until.isoformat() if state.frozen_until else None,
        }
    
    def get_risk_level(self) -> str:
        """Obtiene nivel de riesgo legible."""
        dd = self.state.current_drawdown
        
        if dd >= self._drawdown_freeze_level:
            return "CRITICAL"
        elif dd >= self._drawdown_level_2:
            return "HIGH"
        elif dd >= self._drawdown_level_1:
            return "ELEVATED"
        else:
            return "NORMAL"
