"""
Pump Model v2.0 para estimar P_pump y Expected G.
v2.0: G dinámico basado en datos históricos + regresión incremental.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

from src.storage import DatabaseManager
from config.settings import SETTINGS


@dataclass
class PumpModelParams:
    """
    Parámetros del modelo de pump.
    v2.0: Nombres alineados con trainer para recalibración.
    """
    intercept: float = -0.5
    
    volume_momentum: float = 0.8
    price_momentum: float = 0.6
    tx_momentum: float = 0.4
    buy_pressure: float = 1.2
    liquidity_quality: float = 0.5
    opportunity_score: float = 1.0
    log_liquidity: float = 0.1
    volatility_24h: float = 0.3
    momentum_score: float = 0.5
    
    base_pump_target: float = 0.30
    liquidity_factor_scale: float = 0.5
    volatility_factor_scale: float = 1.5
    
    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PumpModelParams":
        mapped = {}
        for k, v in d.items():
            if k.startswith("coef_"):
                new_k = k.replace("coef_", "")
                if hasattr(cls, new_k):
                    mapped[new_k] = v
            elif hasattr(cls, k):
                mapped[k] = v
        return cls(**mapped)


def sigmoid(x: float) -> float:
    """Función sigmoide con clipping."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


class PumpModel:
    """
    Modelo v2.0 para estimar probabilidad de pump y magnitud esperada.
    
    P_pump = sigmoid(γ0 + Σ γi * xi)
    G = v2.0: MFE promedio histórico por bucket de EVS (o fallback heurístico)
    
    v2.0 Changes:
    - Integración con trainer incremental
    - G dinámico basado en datos históricos
    - Nombres de params alineados con trainer
    """
    
    MODEL_NAME = "pump_model"
    
    FEATURE_NAMES = [
        "volume_momentum",
        "price_momentum",
        "tx_momentum",
        "buy_pressure",
        "liquidity_quality",
        "opportunity_score",
        "log_liquidity",
        "volatility_24h",
        "momentum_score",
    ]
    
    def __init__(self, params: Optional[PumpModelParams] = None):
        self.params = params or PumpModelParams()
        self._db: Optional[DatabaseManager] = None
        self._trainer = None
        self._g_buckets: Dict[str, float] = {}
    
    def set_database(self, db: DatabaseManager) -> None:
        """Configura conexión a base de datos."""
        self._db = db
    
    def get_trainer(self):
        """Obtiene el trainer para recalibración incremental."""
        if self._trainer is None:
            from src.calibration.logistic_trainer import PumpModelTrainer
            self._trainer = PumpModelTrainer(prior_params=self.params.to_dict())
        return self._trainer
    
    async def load_params(self) -> None:
        """Carga parámetros desde la base de datos."""
        if not self._db:
            return
        
        saved_params = await self._db.get_model_params(self.MODEL_NAME)
        if saved_params:
            self.params = PumpModelParams.from_dict(saved_params)
            if self._trainer:
                self._trainer.set_params(self.params.to_dict())
            logger.info(f"Loaded {self.MODEL_NAME} params from database")
        
        g_buckets = await self._db.get_g_buckets()
        for bucket in g_buckets:
            self._g_buckets[bucket["bucket_id"]] = bucket["avg_mfe"]
        if self._g_buckets:
            logger.info(f"Loaded {len(self._g_buckets)} G buckets from database")
    
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
            self.params = PumpModelParams.from_dict(trained_params)
    
    def set_g_buckets(self, buckets: Dict[str, float]) -> None:
        """v2.0: Establece los buckets de G histórico."""
        self._g_buckets = buckets
    
    def calculate_p_pump_linear(self, features: Dict[str, float]) -> float:
        """
        Calcula el score lineal para P_pump.
        v2.0: Usa nombres de params alineados con trainer.
        """
        p = self.params
        
        score = p.intercept
        
        vol_momentum = features.get("volume_momentum", 1)
        vol_momentum_norm = np.tanh((vol_momentum - 1) / 2)
        score += p.volume_momentum * vol_momentum_norm
        
        price_mom = features.get("price_momentum", 1)
        price_momentum_norm = np.tanh((price_mom - 1) / 2)
        score += p.price_momentum * price_momentum_norm
        
        tx_mom = features.get("tx_momentum", 1)
        tx_momentum_norm = np.tanh((tx_mom - 1) / 2)
        score += p.tx_momentum * tx_momentum_norm
        
        buy_pres = features.get("buy_pressure", 0.5)
        buy_pressure_centered = buy_pres - 0.5
        score += p.buy_pressure * buy_pressure_centered
        
        liq_score = features.get("liquidity_quality", features.get("liquidity_score", 0.5))
        score += p.liquidity_quality * liq_score
        
        opp_score = features.get("opportunity_score", 0.5)
        score += p.opportunity_score * opp_score
        
        log_liq = features.get("log_liquidity", 10)
        log_liq_norm = (log_liq - 10) / 5
        score += p.log_liquidity * log_liq_norm
        
        vol_24h = features.get("volatility_24h", 0.1)
        score += p.volatility_24h * vol_24h
        
        mom_score = features.get("momentum_score", 0.5)
        score += p.momentum_score * mom_score
        
        return score
    
    def predict_p_pump(self, features: Dict[str, float]) -> float:
        """
        Predice probabilidad de pump favorable.
        
        Returns:
            P_pump en [0, 1]
        """
        linear_score = self.calculate_p_pump_linear(features)
        return sigmoid(linear_score)
    
    def _get_bucket_for_evs(self, evs_adj: float) -> Optional[str]:
        """v2.0: Determina el bucket de G para un EVS dado."""
        buckets = SETTINGS.get("g_buckets", [])
        for bucket in buckets:
            if bucket["min"] <= evs_adj < bucket["max"]:
                return bucket["id"]
        return None
    
    def calculate_historical_g(self, evs_adj: float) -> Optional[float]:
        """
        v2.0: Calcula G basado en MFE histórico por bucket.
        
        Returns:
            G histórico o None si no hay datos suficientes
        """
        bucket_id = self._get_bucket_for_evs(evs_adj)
        if bucket_id and bucket_id in self._g_buckets:
            return self._g_buckets[bucket_id]
        return None
    
    def calculate_expected_g(
        self, 
        features: Dict[str, float],
        evs_adj: Optional[float] = None,
    ) -> float:
        """
        Calcula Expected Pump Magnitude (G).
        
        v2.0: Primero intenta usar G histórico basado en EVS bucket.
        Si no hay datos, usa cálculo heurístico.
        
        Returns:
            G como porcentaje (0.30 = 30%)
        """
        if evs_adj is not None:
            historical_g = self.calculate_historical_g(evs_adj)
            if historical_g is not None:
                return max(0.10, min(historical_g, 2.0))
        
        p = self.params
        base_g = SETTINGS.get("g_fallback", p.base_pump_target)
        
        log_liq = features.get("log_liquidity", 10)
        liq_factor = 1 + p.liquidity_factor_scale * np.tanh((log_liq - 11) / 3)
        
        vol_24h = features.get("volatility_24h", 0.1)
        vol_factor = 1 + p.volatility_factor_scale * vol_24h
        
        momentum = features.get("momentum_score", 0.5)
        momentum_factor = 0.7 + 0.6 * momentum
        
        max_pump_24h = features.get("max_pump_24h", 0)
        if max_pump_24h > 0:
            historical_factor = min(1 + max_pump_24h * 0.3, 2.0)
        else:
            historical_factor = 1.0
        
        g = base_g * liq_factor * vol_factor * momentum_factor * historical_factor
        g = max(0.10, min(g, 2.0))
        
        return g
    
    def predict(
        self, 
        features: Dict[str, float],
        evs_adj: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Predice P_pump y G.
        
        Args:
            features: Features del token
            evs_adj: v2.0 - EVS_adj para lookup de G histórico
        
        Returns:
            Dict con p_pump y expected_g
        """
        p_pump = self.predict_p_pump(features)
        expected_g = self.calculate_expected_g(features, evs_adj)
        
        return {
            "p_pump": p_pump,
            "expected_g": expected_g,
        }
    
    def predict_with_details(
        self, 
        features: Dict[str, float],
        evs_adj: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predicción con detalles de contribución. v2.0: nombres actualizados."""
        p = self.params
        
        contributions = {
            "intercept": p.intercept,
            "volume_momentum": p.volume_momentum * np.tanh((features.get("volume_momentum", 1) - 1) / 2),
            "price_momentum": p.price_momentum * np.tanh((features.get("price_momentum", 1) - 1) / 2),
            "buy_pressure": p.buy_pressure * (features.get("buy_pressure", 0.5) - 0.5),
            "liquidity": p.liquidity_quality * features.get("liquidity_quality", features.get("liquidity_score", 0.5)),
            "opportunity": p.opportunity_score * features.get("opportunity_score", 0.5),
        }
        
        linear_score = sum(contributions.values())
        p_pump = sigmoid(linear_score)
        expected_g = self.calculate_expected_g(features, evs_adj)
        
        g_source = "historical" if (evs_adj and self.calculate_historical_g(evs_adj)) else "heuristic"
        
        return {
            "p_pump": p_pump,
            "expected_g": expected_g,
            "g_source": g_source,
            "linear_score": linear_score,
            "contributions": contributions,
            "top_factors": sorted(
                contributions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
        }
    
    def batch_predict(self, features_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Predice para múltiples tokens."""
        return [self.predict(f) for f in features_list]
    
    async def update_g_buckets(self, trade_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        v2.0: Actualiza los buckets de G basado en MFE histórico.
        
        Agrupa trades por rango de EVS y calcula MFE promedio.
        """
        buckets_config = SETTINGS.get("g_buckets", [])
        min_samples = SETTINGS.get("g_min_samples", 5)
        
        bucket_data = {b["id"]: [] for b in buckets_config}
        
        for tf in trade_features:
            evs_adj = tf.get("evs_adj", 0)
            mfe = tf.get("actual_mfe", 0)
            
            if mfe is None or mfe <= 0:
                continue
            
            for bucket in buckets_config:
                if bucket["min"] <= evs_adj < bucket["max"]:
                    bucket_data[bucket["id"]].append(mfe)
                    break
        
        updated_buckets = {}
        for bucket in buckets_config:
            bucket_id = bucket["id"]
            mfes = bucket_data[bucket_id]
            
            if len(mfes) >= min_samples:
                avg_mfe = np.mean(mfes)
                updated_buckets[bucket_id] = avg_mfe
                
                if self._db:
                    await self._db.update_g_bucket(
                        bucket_id=bucket_id,
                        evs_min=bucket["min"],
                        evs_max=bucket["max"],
                        avg_mfe=avg_mfe,
                        sample_count=len(mfes),
                    )
        
        self._g_buckets.update(updated_buckets)
        logger.info(f"Updated {len(updated_buckets)} G buckets")
        
        return {
            "updated": list(updated_buckets.keys()),
            "values": updated_buckets,
            "sample_counts": {k: len(v) for k, v in bucket_data.items()},
        }
    
    def update_params(
        self,
        outcomes: List[Dict[str, Any]],
        learning_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Actualiza parámetros con outcomes observados.
        DEPRECATED en v2.0: Usar trainer.train() en su lugar.
        """
        logger.warning("update_params is deprecated in v2.0, use trainer.train() instead")
        return {}
    
    def get_opportunity_assessment(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Genera evaluación de oportunidad legible.
        """
        prediction = self.predict_with_details(features)
        p_pump = prediction["p_pump"]
        expected_g = prediction["expected_g"]
        
        if p_pump > 0.6 and expected_g > 0.25:
            assessment = "STRONG"
        elif p_pump > 0.4 and expected_g > 0.20:
            assessment = "MODERATE"
        elif p_pump > 0.3:
            assessment = "WEAK"
        else:
            assessment = "POOR"
        
        return {
            "assessment": assessment,
            "p_pump": f"{p_pump:.1%}",
            "expected_gain": f"{expected_g:.1%}",
            "confidence": "HIGH" if p_pump > 0.5 else "LOW",
            "key_drivers": [f[0] for f in prediction["top_factors"] if f[1] > 0.1],
        }
