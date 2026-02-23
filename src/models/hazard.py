"""
Hazard Model v2.0 para estimar P_rug (probabilidad de rug pull).
v2.0: Integración con regresión logística incremental.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger

from src.storage import DatabaseManager


@dataclass
class HazardModelParams:
    """
    Parámetros del modelo de hazard.
    v2.0: Estos son los valores por defecto que sirven como prior
    cuando no hay suficientes datos para entrenar.
    """
    intercept: float = -1.5
    
    holder_concentration: float = 2.0
    top_holder_pct: float = 1.5
    liquidity_ratio: float = -3.0
    age_hours: float = -0.1
    has_renounced: float = -1.0
    has_verified: float = -0.5
    has_freeze_auth: float = 0.8
    has_mint_auth: float = 0.6
    volume_liquidity_ratio: float = -0.3
    structural_risk: float = 1.5
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "intercept": self.intercept,
            "holder_concentration": self.holder_concentration,
            "top_holder_pct": self.top_holder_pct,
            "liquidity_ratio": self.liquidity_ratio,
            "age_hours": self.age_hours,
            "has_renounced": self.has_renounced,
            "has_verified": self.has_verified,
            "has_freeze_auth": self.has_freeze_auth,
            "has_mint_auth": self.has_mint_auth,
            "volume_liquidity_ratio": self.volume_liquidity_ratio,
            "structural_risk": self.structural_risk,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "HazardModelParams":
        mapped = {}
        for k, v in d.items():
            if k.startswith("coef_"):
                new_k = k.replace("coef_", "")
                if new_k == "holder_concentration":
                    mapped["holder_concentration"] = v
                elif new_k == "top_holder":
                    mapped["top_holder_pct"] = v
                elif new_k == "age_penalty":
                    mapped["age_hours"] = v
                elif new_k == "renounced_bonus":
                    mapped["has_renounced"] = v
                elif new_k == "verified_bonus":
                    mapped["has_verified"] = v
                elif new_k == "freeze_auth_penalty":
                    mapped["has_freeze_auth"] = v
                elif new_k == "mint_auth_penalty":
                    mapped["has_mint_auth"] = v
                elif new_k == "volume_ratio":
                    mapped["volume_liquidity_ratio"] = v
                else:
                    mapped[new_k] = v
            elif hasattr(cls, k):
                mapped[k] = v
        return cls(**mapped)


def sigmoid(x: float) -> float:
    """Función sigmoide con clipping para evitar overflow."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


class HazardModel:
    """
    Modelo v2.0 para estimar probabilidad de rug pull.
    
    P_rug = sigmoid(β0 + Σ βi * xi)
    
    v2.0 Changes:
    - Integración con IncrementalLogisticRegressor para recalibración
    - Parámetros actualizables via trainer
    - Nombres de features alineados con trainer
    
    Factores de riesgo:
    - Alta concentración de holders (+)
    - Baja liquidez relativa (+)
    - Token muy nuevo (+)
    - Ownership no renunciado (+)
    - Freeze/Mint authority activos (+)
    """
    
    MODEL_NAME = "hazard_model"
    
    FEATURE_NAMES = [
        "holder_concentration",
        "top_holder_pct",
        "liquidity_ratio",
        "age_hours",
        "has_renounced",
        "has_verified",
        "has_freeze_auth",
        "has_mint_auth",
        "volume_liquidity_ratio",
        "structural_risk",
    ]
    
    def __init__(self, params: Optional[HazardModelParams] = None):
        self.params = params or HazardModelParams()
        self._db: Optional[DatabaseManager] = None
        self._trainer = None
    
    def set_database(self, db: DatabaseManager) -> None:
        """Configura conexión a base de datos para persistencia."""
        self._db = db
    
    def get_trainer(self):
        """Obtiene el trainer para recalibración incremental."""
        if self._trainer is None:
            from src.calibration.logistic_trainer import HazardModelTrainer
            self._trainer = HazardModelTrainer(prior_params=self.params.to_dict())
        return self._trainer
    
    async def load_params(self) -> None:
        """Carga parámetros desde la base de datos."""
        if not self._db:
            return
        
        saved_params = await self._db.get_model_params(self.MODEL_NAME)
        if saved_params:
            self.params = HazardModelParams.from_dict(saved_params)
            if self._trainer:
                self._trainer.set_params(self.params.to_dict())
            logger.info(f"Loaded {self.MODEL_NAME} params from database")
    
    async def save_params(self) -> None:
        """Guarda parámetros en la base de datos."""
        if not self._db:
            return
        
        await self._db.save_model_params(self.MODEL_NAME, self.params.to_dict())
        logger.info(f"Saved {self.MODEL_NAME} params to database")
    
    def update_from_trainer(self) -> None:
        """v2.0: Actualiza params desde el trainer después de recalibración."""
        if self._trainer:
            trained_params = self._trainer.get_params()
            self.params = HazardModelParams.from_dict(trained_params)
    
    def calculate_linear_score(self, features: Dict[str, float]) -> float:
        """
        Calcula el score lineal antes de aplicar sigmoid.
        v2.0: Usa nombres de features alineados con trainer.
        """
        p = self.params
        
        score = p.intercept
        
        holder_conc = features.get("holder_concentration", 0.5)
        score += p.holder_concentration * holder_conc
        
        top_holder = features.get("top_holder_pct", 0.2)
        score += p.top_holder_pct * top_holder
        
        liq_ratio = features.get("liquidity_ratio", 0.1)
        score += p.liquidity_ratio * liq_ratio
        
        age_hours_val = features.get("age_hours", 1)
        age_factor = min(age_hours_val / 24, 1)
        score += p.age_hours * age_factor
        
        renounced = features.get("has_renounced", 0)
        score += p.has_renounced * renounced
        
        verified = features.get("has_verified", 0)
        score += p.has_verified * verified
        
        freeze_auth = features.get("has_freeze_auth", 0)
        score += p.has_freeze_auth * freeze_auth
        
        mint_auth = features.get("has_mint_auth", 0)
        score += p.has_mint_auth * mint_auth
        
        vol_liq_ratio = features.get("volume_liquidity_ratio", 1)
        vol_factor = np.tanh(vol_liq_ratio / 3)
        score += p.volume_liquidity_ratio * vol_factor
        
        structural_risk_val = features.get("structural_risk", 0.5)
        score += p.structural_risk * structural_risk_val
        
        return score
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Predice P_rug para un token dado sus features.
        
        Args:
            features: Diccionario de features extraídas
            
        Returns:
            Probabilidad de rug pull [0, 1]
        """
        linear_score = self.calculate_linear_score(features)
        p_rug = sigmoid(linear_score)
        
        return p_rug
    
    def predict_with_details(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predice P_rug con detalles de contribución de cada factor.
        v2.0: Nombres de params actualizados.
        """
        p = self.params
        
        contributions = {
            "intercept": p.intercept,
            "holder_concentration": p.holder_concentration * features.get("holder_concentration", 0.5),
            "top_holder": p.top_holder_pct * features.get("top_holder_pct", 0.2),
            "liquidity_ratio": p.liquidity_ratio * features.get("liquidity_ratio", 0.1),
            "age": p.age_hours * min(features.get("age_hours", 1) / 24, 1),
            "renounced": p.has_renounced * features.get("has_renounced", 0),
            "verified": p.has_verified * features.get("has_verified", 0),
            "freeze_auth": p.has_freeze_auth * features.get("has_freeze_auth", 0),
            "mint_auth": p.has_mint_auth * features.get("has_mint_auth", 0),
        }
        
        linear_score = sum(contributions.values())
        p_rug = sigmoid(linear_score)
        
        return {
            "p_rug": p_rug,
            "linear_score": linear_score,
            "contributions": contributions,
            "top_risk_factors": sorted(
                contributions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3],
        }
    
    def batch_predict(self, features_list: List[Dict[str, float]]) -> List[float]:
        """Predice P_rug para múltiples tokens."""
        return [self.predict(f) for f in features_list]
    
    def update_params(
        self, 
        outcomes: List[Dict[str, Any]], 
        learning_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Actualiza parámetros basándose en outcomes observados.
        
        Usa gradient descent simplificado para ajustar coeficientes.
        
        Args:
            outcomes: Lista de {features, is_rug} observados
            learning_rate: Tasa de aprendizaje
            
        Returns:
            Cambios aplicados a cada parámetro
        """
        if not outcomes:
            return {}
        
        param_updates = {k: 0.0 for k in self.params.to_dict().keys()}
        
        for outcome in outcomes:
            features = outcome["features"]
            is_rug = outcome["is_rug"]
            
            p_rug = self.predict(features)
            error = is_rug - p_rug
            
            gradient = error * p_rug * (1 - p_rug)
            
            param_updates["intercept"] += learning_rate * gradient
            param_updates["coef_holder_concentration"] += (
                learning_rate * gradient * features.get("holder_concentration", 0.5)
            )
            param_updates["coef_liquidity_ratio"] += (
                learning_rate * gradient * features.get("liquidity_ratio", 0.1)
            )
        
        n = len(outcomes)
        changes = {}
        
        for param_name, update in param_updates.items():
            avg_update = update / n
            old_value = getattr(self.params, param_name)
            new_value = old_value + avg_update
            setattr(self.params, param_name, new_value)
            changes[param_name] = avg_update
        
        logger.info(f"Updated hazard model with {n} outcomes")
        
        return changes
    
    def get_risk_breakdown(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Genera un desglose legible del riesgo.
        """
        details = self.predict_with_details(features)
        p_rug = details["p_rug"]
        
        if p_rug < 0.2:
            risk_level = "LOW"
        elif p_rug < 0.4:
            risk_level = "MODERATE"
        elif p_rug < 0.6:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"
        
        breakdown = {
            "risk_level": risk_level,
            "p_rug": f"{p_rug:.1%}",
            "main_factors": [],
        }
        
        for factor, contribution in details["top_risk_factors"]:
            if contribution > 0.3:
                breakdown["main_factors"].append(f"{factor}: HIGH RISK")
            elif contribution > 0.1:
                breakdown["main_factors"].append(f"{factor}: moderate risk")
        
        return breakdown
