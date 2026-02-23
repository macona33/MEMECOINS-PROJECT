"""
EVS Calculator - Expected Value Score y Risk-Adjusted EVS.
Combina Hazard y Pump models para scoring final.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger

from .hazard import HazardModel
from .pump import PumpModel
from src.storage import DatabaseManager
from config.settings import SETTINGS


@dataclass
class EVSResult:
    """Resultado del cálculo de EVS."""
    evs: float
    evs_adj: float
    p_rug: float
    p_pump: float
    expected_g: float
    sigma_token: float
    components: Dict[str, float]
    is_tradeable: bool
    rejection_reason: Optional[str] = None


class EVSCalculator:
    """
    Calculador de Expected Value Score.
    
    EVS = P_pump * G - P_rug * L - costs
    EVS_adj = EVS / (σ_token + ε)
    
    Donde:
    - P_pump: Probabilidad de pump (del PumpModel)
    - G: Expected pump magnitude
    - P_rug: Probabilidad de rug (del HazardModel)
    - L: Pérdida esperada en rug (~95%)
    - costs: Slippage + fees
    - σ_token: Volatilidad del token
    - ε: Constante de suavizado
    """
    
    def __init__(
        self,
        hazard_model: Optional[HazardModel] = None,
        pump_model: Optional[PumpModel] = None,
    ):
        self.hazard_model = hazard_model or HazardModel()
        self.pump_model = pump_model or PumpModel()
        
        self.rug_loss = SETTINGS["rug_loss_pct"]
        self.slippage = SETTINGS["slippage_pct"]
        self.fees = SETTINGS["fees_pct"]
        self.epsilon = 0.01
        
        self.evs_threshold = SETTINGS["evs_adj_threshold"]
        self.max_p_rug = SETTINGS["max_p_rug"]
        self.min_p_pump = SETTINGS["min_p_pump"]
    
    def set_database(self, db: DatabaseManager) -> None:
        """Configura base de datos para los modelos."""
        self.hazard_model.set_database(db)
        self.pump_model.set_database(db)
    
    async def load_models(self) -> None:
        """Carga parámetros de ambos modelos."""
        await self.hazard_model.load_params()
        await self.pump_model.load_params()
    
    async def save_models(self) -> None:
        """Guarda parámetros de ambos modelos."""
        await self.hazard_model.save_params()
        await self.pump_model.save_params()
    
    def calculate_costs(self, features: Dict[str, float]) -> float:
        """
        Calcula costes totales estimados.
        Incluye slippage dinámico basado en liquidez.
        """
        base_costs = self.slippage + self.fees
        
        log_liq = features.get("log_liquidity", 10)
        if log_liq < 9:
            slippage_adj = (9 - log_liq) * 0.01
            base_costs += slippage_adj
        
        vol = features.get("volatility_24h", 0.1)
        if vol > 0.2:
            volatility_cost = (vol - 0.2) * 0.5
            base_costs += volatility_cost
        
        return min(base_costs, 0.15)
    
    def estimate_sigma(self, features: Dict[str, float]) -> float:
        """
        Estima volatilidad del token.
        Usa datos históricos si disponibles, sino estima.
        """
        vol_24h = features.get("volatility_24h")
        if vol_24h is not None and vol_24h > 0:
            return vol_24h
        
        estimated_vol = 0.10
        
        price_change_1h = abs(features.get("price_change_1h", 0))
        if price_change_1h > 0:
            estimated_vol = max(estimated_vol, price_change_1h * 2)
        
        vol_momentum = features.get("volume_momentum", 1)
        if vol_momentum > 2:
            estimated_vol *= 1.5
        
        return min(estimated_vol, 0.50)
    
    def calculate(self, features: Dict[str, float]) -> EVSResult:
        """
        Calcula EVS y EVS_adj para un token.
        
        Args:
            features: Features extraídas del token
            
        Returns:
            EVSResult con todos los componentes
        """
        p_rug = self.hazard_model.predict(features)
        
        pump_result = self.pump_model.predict(features)
        p_pump = pump_result["p_pump"]
        expected_g = pump_result["expected_g"]
        
        costs = self.calculate_costs(features)
        sigma = self.estimate_sigma(features)
        
        expected_gain = p_pump * expected_g
        expected_loss = p_rug * self.rug_loss
        evs = expected_gain - expected_loss - costs
        
        evs_adj = evs / (sigma + self.epsilon)
        
        is_tradeable, rejection = self._check_tradeable(p_rug, p_pump, evs_adj)
        
        components = {
            "expected_gain": expected_gain,
            "expected_loss": expected_loss,
            "costs": costs,
            "p_pump_component": p_pump * expected_g,
            "p_rug_component": p_rug * self.rug_loss,
        }
        
        return EVSResult(
            evs=evs,
            evs_adj=evs_adj,
            p_rug=p_rug,
            p_pump=p_pump,
            expected_g=expected_g,
            sigma_token=sigma,
            components=components,
            is_tradeable=is_tradeable,
            rejection_reason=rejection,
        )
    
    def _check_tradeable(
        self, 
        p_rug: float, 
        p_pump: float, 
        evs_adj: float
    ) -> Tuple[bool, Optional[str]]:
        """Verifica si token cumple criterios para trading."""
        if p_rug > self.max_p_rug:
            return False, f"P_rug too high: {p_rug:.1%} > {self.max_p_rug:.1%}"
        
        if p_pump < self.min_p_pump:
            return False, f"P_pump too low: {p_pump:.1%} < {self.min_p_pump:.1%}"
        
        if evs_adj < self.evs_threshold:
            return False, f"EVS_adj too low: {evs_adj:.3f} < {self.evs_threshold}"
        
        return True, None
    
    def calculate_with_details(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Calcula EVS con detalles completos para análisis.
        """
        result = self.calculate(features)
        
        hazard_details = self.hazard_model.predict_with_details(features)
        pump_details = self.pump_model.predict_with_details(features)
        
        return {
            "evs": result.evs,
            "evs_adj": result.evs_adj,
            "p_rug": result.p_rug,
            "p_pump": result.p_pump,
            "expected_g": result.expected_g,
            "sigma_token": result.sigma_token,
            "is_tradeable": result.is_tradeable,
            "rejection_reason": result.rejection_reason,
            "components": result.components,
            "hazard_details": hazard_details,
            "pump_details": pump_details,
        }
    
    def batch_calculate(
        self, 
        features_list: List[Dict[str, float]]
    ) -> List[EVSResult]:
        """Calcula EVS para múltiples tokens."""
        return [self.calculate(f) for f in features_list]
    
    def rank_tokens(
        self, 
        tokens_features: List[Tuple[str, Dict[str, float]]]
    ) -> List[Tuple[str, EVSResult, int]]:
        """
        Rankea tokens por EVS_adj.
        
        Args:
            tokens_features: Lista de (address, features)
            
        Returns:
            Lista de (address, EVSResult, rank) ordenada por EVS_adj
        """
        results = []
        
        for address, features in tokens_features:
            evs_result = self.calculate(features)
            results.append((address, evs_result))
        
        results.sort(key=lambda x: x[1].evs_adj, reverse=True)
        
        ranked = [
            (address, result, rank + 1)
            for rank, (address, result) in enumerate(results)
        ]
        
        return ranked
    
    def get_tradeable_tokens(
        self, 
        tokens_features: List[Tuple[str, Dict[str, float]]],
        max_tokens: int = 10
    ) -> List[Tuple[str, EVSResult]]:
        """
        Obtiene tokens tradeables ordenados por EVS_adj.
        """
        ranked = self.rank_tokens(tokens_features)
        
        tradeable = [
            (addr, result)
            for addr, result, _ in ranked
            if result.is_tradeable
        ]
        
        return tradeable[:max_tokens]
    
    def generate_report(self, features: Dict[str, float], token_info: Dict[str, Any] = None) -> str:
        """
        Genera reporte legible del análisis EVS.
        """
        result = self.calculate_with_details(features)
        
        symbol = token_info.get("symbol", "TOKEN") if token_info else "TOKEN"
        
        lines = [
            f"=== EVS Analysis: {symbol} ===",
            "",
            f"Expected Value Score: {result['evs']:.4f}",
            f"Risk-Adjusted EVS:    {result['evs_adj']:.4f}",
            "",
            "--- Risk Assessment ---",
            f"P(Rug Pull):  {result['p_rug']:.1%}",
            f"P(Pump):      {result['p_pump']:.1%}",
            f"Expected G:   {result['expected_g']:.1%}",
            f"Volatility:   {result['sigma_token']:.1%}",
            "",
            "--- Components ---",
            f"Expected Gain:  +{result['components']['expected_gain']:.2%}",
            f"Expected Loss:  -{result['components']['expected_loss']:.2%}",
            f"Trading Costs:  -{result['components']['costs']:.2%}",
            "",
            "--- Decision ---",
        ]
        
        if result["is_tradeable"]:
            lines.append("STATUS: TRADEABLE")
        else:
            lines.append(f"STATUS: REJECTED - {result['rejection_reason']}")
        
        return "\n".join(lines)
    
    async def save_scores(
        self, 
        db: DatabaseManager,
        address: str,
        features: Dict[str, float],
        rank: int = None
    ) -> EVSResult:
        """
        Calcula y guarda scores en la base de datos.
        """
        result = self.calculate(features)
        
        scores = {
            "p_rug": result.p_rug,
            "p_pump": result.p_pump,
            "expected_g": result.expected_g,
            "evs": result.evs,
            "evs_adj": result.evs_adj,
            "sigma_token": result.sigma_token,
            "rank": rank,
        }
        
        await db.insert_scores(address, scores)
        
        return result
