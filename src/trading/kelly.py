"""
Kelly Criterion v2.0 para position sizing.
v2.0: Gamma conservador (0.10), caps por volatilidad y etapa temprana.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from config.settings import SETTINGS


@dataclass
class PositionSize:
    """Resultado del cálculo de position size."""
    kelly_fraction: float
    position_pct: float
    position_usd: float
    rationale: str
    
    @property
    def is_valid(self) -> bool:
        return self.kelly_fraction > 0 and self.position_usd > 0


class KellyCalculator:
    """
    Calculador de Kelly Criterion fraccional v2.0.
    
    f* = (p * b - q) / b  (Kelly clásico)
    f = γ * EVS_adj / Var  (Ajustado por volatilidad)
    
    v2.0 Changes:
    - γ reducido a 0.10 (era 0.25)
    - max_position_pct = 0.03 mientras dataset < 100 trades
    - Cap adicional si sigma < 0.05 (max 2%)
    - Integración con gamma_multiplier del RiskManager
    
    Donde:
    - γ: Fracción de Kelly (0.10 en v2.0)
    - EVS_adj: Expected Value Score ajustado
    - Var: Varianza estimada
    """
    
    def __init__(
        self,
        gamma: float = None,
        max_position_pct: float = None,
        min_position_usd: float = 50,
        max_position_usd: float = None,
    ):
        self.gamma = gamma or SETTINGS["kelly_gamma"]  # v2.0: 0.10
        self.max_position_pct = max_position_pct or SETTINGS["max_position_pct"]  # v2.0: 0.03
        self.min_position_usd = min_position_usd
        self.max_position_usd = max_position_usd
        self.capital = SETTINGS["initial_capital"]
        
        self._gamma_multiplier = 1.0
        self._trade_count = 0
        
        self._min_sigma_cap = SETTINGS.get("min_sigma_cap", 0.05)
        self._low_sigma_max_kelly = SETTINGS.get("low_sigma_max_kelly", 0.02)
        self._min_trades_for_full_kelly = SETTINGS.get("min_trades_for_full_kelly", 100)
        self._early_stage_max_position = SETTINGS.get("early_stage_max_position", 0.03)
        self._mature_stage_max_position = SETTINGS.get("mature_stage_max_position", 0.05)
    
    def set_capital(self, capital: float) -> None:
        """Actualiza el capital disponible."""
        self.capital = capital
    
    def set_gamma_multiplier(self, multiplier: float) -> None:
        """v2.0: Establece multiplicador de gamma desde RiskManager."""
        self._gamma_multiplier = max(0.1, min(multiplier, 1.0))
        logger.debug(f"Kelly gamma multiplier set to {self._gamma_multiplier:.2f}")
    
    def set_trade_count(self, count: int) -> None:
        """v2.0: Establece número de trades para ajustar max_position."""
        self._trade_count = count
        if count >= self._min_trades_for_full_kelly:
            self.max_position_pct = self._mature_stage_max_position
        else:
            self.max_position_pct = self._early_stage_max_position
    
    def calculate_classic_kelly(
        self,
        win_prob: float,
        win_ratio: float,
        loss_ratio: float = 1.0
    ) -> float:
        """
        Calcula Kelly clásico.
        
        f* = (p * b - q) / b
        
        Args:
            win_prob: Probabilidad de ganar (p)
            win_ratio: Ratio de ganancia si ganas (b)
            loss_ratio: Ratio de pérdida si pierdes (por defecto 1 = 100%)
            
        Returns:
            Fracción óptima de Kelly
        """
        if win_ratio <= 0:
            return 0.0
        
        p = win_prob
        q = 1 - p
        b = win_ratio / loss_ratio
        
        kelly = (p * b - q) / b
        
        return max(0, kelly)
    
    def calculate_evs_kelly(
        self,
        evs_adj: float,
        sigma: float,
        p_rug: float = 0.0
    ) -> float:
        """
        Calcula Kelly basado en EVS ajustado.
        
        f = γ * gamma_mult * EVS_adj / (σ² + EVS_adj²)
        
        v2.0 Changes:
        - Aplica gamma_multiplier del RiskManager
        - Cap adicional si sigma < min_sigma_cap
        
        Args:
            evs_adj: Expected Value Score ajustado por riesgo
            sigma: Volatilidad del token
            p_rug: Probabilidad de rug (para penalización adicional)
            
        Returns:
            Fracción de Kelly ajustada
        """
        if evs_adj <= 0:
            return 0.0
        
        effective_gamma = self.gamma * self._gamma_multiplier
        
        variance = sigma ** 2
        
        base_kelly = effective_gamma * evs_adj / (variance + evs_adj ** 2 + 0.01)
        
        if p_rug > 0.2:
            rug_penalty = 1 - (p_rug - 0.2) * 2
            base_kelly *= max(0.3, rug_penalty)
        
        if sigma < self._min_sigma_cap:
            base_kelly = min(base_kelly, self._low_sigma_max_kelly)
            logger.debug(f"Low sigma cap applied: kelly={base_kelly:.3f}, sigma={sigma:.3f}")
        
        return max(0, base_kelly)
    
    def calculate_position(
        self,
        evs_result: Any,
        features: Dict[str, float] = None,
        available_capital: float = None
    ) -> PositionSize:
        """
        Calcula tamaño de posición completo.
        
        Args:
            evs_result: Resultado del EVSCalculator
            features: Features adicionales del token
            available_capital: Capital disponible (usa self.capital si no se provee)
            
        Returns:
            PositionSize con todos los detalles
        """
        capital = available_capital or self.capital
        
        if not evs_result.is_tradeable:
            return PositionSize(
                kelly_fraction=0,
                position_pct=0,
                position_usd=0,
                rationale=f"Not tradeable: {evs_result.rejection_reason}"
            )
        
        kelly_f = self.calculate_evs_kelly(
            evs_adj=evs_result.evs_adj,
            sigma=evs_result.sigma_token,
            p_rug=evs_result.p_rug
        )
        
        position_pct = min(kelly_f, self.max_position_pct)
        
        position_usd = capital * position_pct
        
        if self.max_position_usd:
            position_usd = min(position_usd, self.max_position_usd)
            position_pct = position_usd / capital
        
        if position_usd < self.min_position_usd:
            return PositionSize(
                kelly_fraction=kelly_f,
                position_pct=0,
                position_usd=0,
                rationale=f"Position too small: ${position_usd:.2f} < ${self.min_position_usd}"
            )
        
        rationale = self._generate_rationale(
            kelly_f, position_pct, evs_result
        )
        
        return PositionSize(
            kelly_fraction=kelly_f,
            position_pct=position_pct,
            position_usd=position_usd,
            rationale=rationale
        )
    
    def _generate_rationale(
        self,
        kelly_f: float,
        position_pct: float,
        evs_result: Any
    ) -> str:
        """Genera explicación del sizing."""
        parts = []
        
        parts.append(f"Kelly={kelly_f:.2%}")
        
        if position_pct < kelly_f:
            parts.append(f"capped to {position_pct:.2%}")
        
        parts.append(f"EVS_adj={evs_result.evs_adj:.3f}")
        parts.append(f"σ={evs_result.sigma_token:.1%}")
        
        return ", ".join(parts)
    
    def adjust_for_correlation(
        self,
        positions: list,
        new_position: PositionSize,
        correlation: float = 0.5
    ) -> PositionSize:
        """
        Ajusta posición considerando correlación con posiciones existentes.
        
        Para memecoins en Solana, asumimos alta correlación entre tokens.
        """
        if not positions:
            return new_position
        
        total_exposure = sum(p.position_pct for p in positions)
        
        max_total_exposure = 0.30
        
        if total_exposure >= max_total_exposure:
            return PositionSize(
                kelly_fraction=new_position.kelly_fraction,
                position_pct=0,
                position_usd=0,
                rationale=f"Max portfolio exposure reached: {total_exposure:.1%}"
            )
        
        available_exposure = max_total_exposure - total_exposure
        
        if new_position.position_pct > available_exposure:
            adjusted_pct = available_exposure
            adjusted_usd = self.capital * adjusted_pct
            
            return PositionSize(
                kelly_fraction=new_position.kelly_fraction,
                position_pct=adjusted_pct,
                position_usd=adjusted_usd,
                rationale=f"Reduced for portfolio diversification: {adjusted_pct:.1%}"
            )
        
        return new_position
    
    def calculate_portfolio_kelly(
        self,
        tokens_evs: list,
        max_positions: int = None
    ) -> list:
        """
        Calcula Kelly óptimo para un portfolio de tokens.
        
        Args:
            tokens_evs: Lista de (address, evs_result, features)
            max_positions: Máximo número de posiciones
            
        Returns:
            Lista de (address, PositionSize)
        """
        max_pos = max_positions or SETTINGS["max_concurrent_trades"]
        
        sorted_tokens = sorted(
            tokens_evs,
            key=lambda x: x[1].evs_adj,
            reverse=True
        )
        
        positions = []
        total_allocated = 0.0
        
        for address, evs_result, features in sorted_tokens[:max_pos]:
            if total_allocated >= 0.30:
                break
            
            position = self.calculate_position(evs_result, features)
            
            if position.is_valid:
                if total_allocated + position.position_pct > 0.30:
                    remaining = 0.30 - total_allocated
                    position = PositionSize(
                        kelly_fraction=position.kelly_fraction,
                        position_pct=remaining,
                        position_usd=self.capital * remaining,
                        rationale=f"Capped for portfolio limit"
                    )
                
                positions.append((address, position))
                total_allocated += position.position_pct
        
        return positions
    
    def calculate_stop_levels(
        self,
        entry_price: float,
        evs_result: Any,
        features: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calcula niveles de stop loss y take profit.
        """
        base_stop = SETTINGS["base_stop_loss"]
        take_profit = SETTINGS["take_profit_target"]
        
        sigma = evs_result.sigma_token
        volatility_adj = min(sigma * 0.5, 0.10)
        
        stop_loss_pct = base_stop + volatility_adj
        
        if evs_result.p_rug > 0.3:
            stop_loss_pct *= 0.8
        
        take_profit_pct = max(take_profit, evs_result.expected_g * 0.8)
        
        return {
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "stop_price": entry_price * (1 - stop_loss_pct),
            "take_profit_price": entry_price * (1 + take_profit_pct),
            "trailing_stop_activation": entry_price * (1 + take_profit_pct * 0.5),
        }
