"""
Market Regime Detector v2.0.
Detecta condiciones de mercado y ajusta parámetros de trading dinámicamente.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from config.settings import SETTINGS
from src.storage import DatabaseManager


class MarketRegime(Enum):
    """Regímenes de mercado posibles."""
    LOW_ACTIVITY = "LOW_ACTIVITY"
    NORMAL = "NORMAL"
    HIGH_ACTIVITY = "HIGH_ACTIVITY"


@dataclass
class RegimeMetrics:
    """Métricas usadas para determinar régimen."""
    new_tokens_2h: int
    total_volume_2h: float
    pct_with_volume: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_tokens_2h": self.new_tokens_2h,
            "total_volume_2h": self.total_volume_2h,
            "pct_with_volume": self.pct_with_volume,
        }


@dataclass
class RegimeAdjustments:
    """Ajustes a aplicar según el régimen."""
    evs_threshold: float
    max_position_pct: float
    max_concurrent_trades: int
    regime: MarketRegime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evs_threshold": self.evs_threshold,
            "max_position_pct": self.max_position_pct,
            "max_concurrent_trades": self.max_concurrent_trades,
            "regime": self.regime.value,
        }


class RegimeDetector:
    """
    Detector de régimen de mercado v2.0.
    
    Analiza condiciones globales del mercado cada ciclo y ajusta:
    - EVS threshold (más alto en baja actividad)
    - Tamaño máximo de posición
    - Número máximo de trades concurrentes
    
    Métricas evaluadas:
    - Número de tokens nuevos en las últimas 2 horas
    - Volumen total agregado
    - Porcentaje de tokens con volumen significativo
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self._db = db
        self._current_regime = MarketRegime.NORMAL
        self._last_check: Optional[datetime] = None
        self._metrics_history: List[RegimeMetrics] = []
        
        self._low_tokens_threshold = SETTINGS.get("regime_low_tokens_threshold", 20)
        self._high_tokens_threshold = SETTINGS.get("regime_high_tokens_threshold", 100)
        self._low_volume_pct = SETTINGS.get("regime_low_volume_pct", 0.30)
        self._high_volume_pct = SETTINGS.get("regime_high_volume_pct", 0.70)
        self._volume_threshold = SETTINGS.get("regime_volume_threshold", 5000)
        self._check_hours = SETTINGS.get("regime_check_hours", 2)
    
    def set_database(self, db: DatabaseManager) -> None:
        """Configura conexión a base de datos."""
        self._db = db
    
    @property
    def current_regime(self) -> MarketRegime:
        """Régimen actual."""
        return self._current_regime
    
    async def load_state(self) -> None:
        """Carga estado de régimen desde la base de datos."""
        if not self._db:
            return
        
        state = await self._db.get_market_regime()
        if state:
            regime_str = state.get("regime", "NORMAL")
            try:
                self._current_regime = MarketRegime(regime_str)
            except ValueError:
                self._current_regime = MarketRegime.NORMAL
            logger.info(f"Loaded market regime: {self._current_regime.value}")
    
    async def calculate_metrics(self, tokens: List[Dict[str, Any]]) -> RegimeMetrics:
        """
        Calcula métricas de mercado a partir de tokens recientes.
        
        Args:
            tokens: Lista de tokens detectados en las últimas horas
            
        Returns:
            RegimeMetrics con métricas calculadas
        """
        cutoff = datetime.now() - timedelta(hours=self._check_hours)
        
        recent_tokens = []
        for token in tokens:
            detected_at = token.get("detected_at")
            if detected_at:
                if isinstance(detected_at, str):
                    try:
                        detected_dt = datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
                        if detected_dt.tzinfo:
                            detected_dt = detected_dt.replace(tzinfo=None)
                    except (ValueError, TypeError):
                        continue
                else:
                    detected_dt = detected_at
                
                if detected_dt >= cutoff:
                    recent_tokens.append(token)
        
        new_tokens_count = len(recent_tokens)
        
        total_volume = sum(
            token.get("volume_24h", 0) or 0 
            for token in recent_tokens
        )
        
        tokens_with_volume = sum(
            1 for token in recent_tokens
            if (token.get("volume_24h", 0) or 0) >= self._volume_threshold
        )
        pct_with_volume = tokens_with_volume / new_tokens_count if new_tokens_count > 0 else 0
        
        metrics = RegimeMetrics(
            new_tokens_2h=new_tokens_count,
            total_volume_2h=total_volume,
            pct_with_volume=pct_with_volume,
            timestamp=datetime.now(),
        )
        
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]
        
        return metrics
    
    def classify_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """
        Clasifica el régimen de mercado basado en métricas.
        
        Criterios:
        - LOW_ACTIVITY: < 20 tokens OR < 30% con volumen
        - HIGH_ACTIVITY: > 100 tokens OR > 70% con volumen
        - NORMAL: resto
        """
        if (metrics.new_tokens_2h < self._low_tokens_threshold or 
            metrics.pct_with_volume < self._low_volume_pct):
            return MarketRegime.LOW_ACTIVITY
        
        if (metrics.new_tokens_2h > self._high_tokens_threshold or 
            metrics.pct_with_volume > self._high_volume_pct):
            return MarketRegime.HIGH_ACTIVITY
        
        return MarketRegime.NORMAL
    
    def get_adjustments(self, regime: Optional[MarketRegime] = None) -> RegimeAdjustments:
        """
        Obtiene ajustes de parámetros para un régimen dado.
        
        Returns:
            RegimeAdjustments con parámetros ajustados
        """
        if regime is None:
            regime = self._current_regime
        
        base_evs = SETTINGS.get("evs_adj_threshold", 0.05)
        base_position = SETTINGS.get("max_position_pct", 0.03)
        base_concurrent = SETTINGS.get("max_concurrent_trades", 10)
        
        adjustments = SETTINGS.get("regime_adjustments", {})
        regime_adj = adjustments.get(regime.value, {})
        
        evs_mult = regime_adj.get("evs_threshold_mult", 1.0)
        pos_mult = regime_adj.get("max_position_mult", 1.0)
        conc_mult = regime_adj.get("max_concurrent_mult", 1.0)
        
        return RegimeAdjustments(
            evs_threshold=base_evs * evs_mult,
            max_position_pct=base_position * pos_mult,
            max_concurrent_trades=int(base_concurrent * conc_mult),
            regime=regime,
        )
    
    async def update(self, tokens: List[Dict[str, Any]]) -> RegimeAdjustments:
        """
        Actualiza el régimen de mercado basado en tokens recientes.
        
        Args:
            tokens: Lista de tokens detectados recientemente
            
        Returns:
            RegimeAdjustments con parámetros actualizados
        """
        metrics = await self.calculate_metrics(tokens)
        new_regime = self.classify_regime(metrics)
        
        if new_regime != self._current_regime:
            logger.info(
                f"Market regime changed: {self._current_regime.value} -> {new_regime.value} "
                f"(tokens={metrics.new_tokens_2h}, vol_pct={metrics.pct_with_volume:.1%})"
            )
            self._current_regime = new_regime
            
            if self._db:
                await self._db.update_market_regime(
                    regime=new_regime.value,
                    metrics=metrics.to_dict(),
                )
        
        self._last_check = datetime.now()
        
        return self.get_adjustments()
    
    def should_check(self) -> bool:
        """Determina si es hora de verificar el régimen."""
        if self._last_check is None:
            return True
        
        check_interval = timedelta(minutes=30)
        return datetime.now() - self._last_check >= check_interval
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del detector."""
        return {
            "regime": self._current_regime.value,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "adjustments": self.get_adjustments().to_dict(),
            "history_length": len(self._metrics_history),
        }
    
    def get_regime_description(self) -> str:
        """Obtiene descripción legible del régimen actual."""
        descriptions = {
            MarketRegime.LOW_ACTIVITY: (
                "Baja actividad - Thresholds elevados, posiciones reducidas"
            ),
            MarketRegime.NORMAL: (
                "Actividad normal - Parámetros base"
            ),
            MarketRegime.HIGH_ACTIVITY: (
                "Alta actividad - Más oportunidades, más trades concurrentes"
            ),
        }
        return descriptions.get(self._current_regime, "Unknown")
