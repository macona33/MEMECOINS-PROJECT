"""
Label Generator v2.0 - Generador de labels path-dependent.
v2.0: Labels alineados con stop_loss ejecutado.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from config.settings import SETTINGS


class TradeLabel(Enum):
    """Labels posibles para un trade."""
    PUMP = "pump"
    RUG = "rug"
    NEUTRAL = "neutral"
    BREAKEVEN = "breakeven"


@dataclass
class LabelResult:
    """Resultado de la clasificación de un trade."""
    label: TradeLabel
    confidence: float
    rationale: str
    metrics: Dict[str, float]


class LabelGenerator:
    """
    Generador de labels path-dependent v2.0.
    
    v2.0 Change: RUG label ahora se alinea con el stop_loss ejecutado,
    no con un threshold separado. Esto asegura que el modelo aprenda
    exactamente lo que se ejecuta.
    
    Criterios:
    - PUMP: MFE >= α (take_profit) antes de activar stop
    - RUG: MAE <= -stop_loss_pct (stop ejecutado)
    - NEUTRAL: No cumple ninguno de los anteriores
    - BREAKEVEN: PnL final cercano a 0
    """
    
    def __init__(
        self,
        pump_threshold: float = None,
        default_stop_loss: float = None,
        breakeven_tolerance: float = 0.02,
    ):
        self.alpha = pump_threshold or SETTINGS["pump_threshold_alpha"]
        self.default_stop_loss = default_stop_loss or SETTINGS["base_stop_loss"]
        self.breakeven_tolerance = breakeven_tolerance
    
    def generate_label(
        self,
        mfe: float,
        mae: float,
        pnl: float,
        mfe_time: Optional[float] = None,
        mae_time: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
    ) -> str:
        """
        Genera label para un trade completado.
        
        Args:
            mfe: Maximum Favorable Excursion (positivo)
            mae: Maximum Adverse Excursion (negativo)
            pnl: PnL final
            mfe_time: Tiempo hasta MFE (minutos)
            mae_time: Tiempo hasta MAE (minutos)
            stop_loss_pct: v2.0 - Stop loss que se usó para este trade
            
        Returns:
            Label string ('pump', 'rug', 'neutral', 'breakeven')
        """
        result = self.generate_label_detailed(mfe, mae, pnl, mfe_time, mae_time, stop_loss_pct)
        return result.label.value
    
    def generate_label_detailed(
        self,
        mfe: float,
        mae: float,
        pnl: float,
        mfe_time: Optional[float] = None,
        mae_time: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
    ) -> LabelResult:
        """
        Genera label con detalles completos.
        
        v2.0: El threshold de RUG ahora usa el stop_loss real del trade,
        no un valor fijo. Esto alinea el modelo con lo que realmente se ejecuta.
        """
        effective_stop = stop_loss_pct if stop_loss_pct is not None else self.default_stop_loss
        
        metrics = {
            "mfe": mfe,
            "mae": mae,
            "pnl": pnl,
            "mfe_time": mfe_time,
            "mae_time": mae_time,
            "stop_loss_used": effective_stop,
        }
        
        is_pump_candidate = mfe >= self.alpha
        is_rug_candidate = mae <= -effective_stop
        
        if is_pump_candidate and not is_rug_candidate:
            return LabelResult(
                label=TradeLabel.PUMP,
                confidence=min(mfe / self.alpha, 1.5) / 1.5,
                rationale=f"MFE {mfe:.1%} >= {self.alpha:.1%}, MAE {mae:.1%} > -{effective_stop:.1%}",
                metrics=metrics,
            )
        
        if is_pump_candidate and is_rug_candidate:
            if mfe_time is not None and mae_time is not None:
                if mfe_time < mae_time:
                    return LabelResult(
                        label=TradeLabel.PUMP,
                        confidence=0.7,
                        rationale=f"MFE reached before stop (t_mfe={mfe_time:.0f}min < t_mae={mae_time:.0f}min)",
                        metrics=metrics,
                    )
                else:
                    return LabelResult(
                        label=TradeLabel.RUG,
                        confidence=0.7,
                        rationale=f"Stop hit before MFE (t_mae={mae_time:.0f}min < t_mfe={mfe_time:.0f}min)",
                        metrics=metrics,
                    )
            
            if pnl > 0:
                return LabelResult(
                    label=TradeLabel.PUMP,
                    confidence=0.5,
                    rationale=f"Both thresholds hit, but positive PnL {pnl:.1%}",
                    metrics=metrics,
                )
            else:
                return LabelResult(
                    label=TradeLabel.RUG,
                    confidence=0.5,
                    rationale=f"Both thresholds hit, negative PnL {pnl:.1%}",
                    metrics=metrics,
                )
        
        if is_rug_candidate:
            return LabelResult(
                label=TradeLabel.RUG,
                confidence=min(abs(mae) / effective_stop, 1.5) / 1.5,
                rationale=f"MAE {mae:.1%} <= -{effective_stop:.1%} (stop executed)",
                metrics=metrics,
            )
        
        if abs(pnl) <= self.breakeven_tolerance:
            return LabelResult(
                label=TradeLabel.BREAKEVEN,
                confidence=1 - abs(pnl) / self.breakeven_tolerance,
                rationale=f"PnL {pnl:.1%} within breakeven tolerance",
                metrics=metrics,
            )
        
        return LabelResult(
            label=TradeLabel.NEUTRAL,
            confidence=0.8,
            rationale=f"MFE {mfe:.1%} < {self.alpha:.1%}, MAE {mae:.1%} > -{effective_stop:.1%}",
            metrics=metrics,
        )
    
    def batch_label(
        self, 
        trades: List[Dict[str, Any]]
    ) -> List[LabelResult]:
        """
        Genera labels para múltiples trades.
        v2.0: Ahora usa stop_loss_pct específico de cada trade.
        """
        results = []
        
        for trade in trades:
            result = self.generate_label_detailed(
                mfe=trade.get("mfe", 0),
                mae=trade.get("mae", 0),
                pnl=trade.get("pnl_pct", 0),
                mfe_time=trade.get("mfe_time"),
                mae_time=trade.get("mae_time"),
                stop_loss_pct=trade.get("stop_loss_pct"),
            )
            results.append(result)
        
        return results
    
    def get_label_distribution(
        self, 
        labels: List[LabelResult]
    ) -> Dict[str, Any]:
        """
        Calcula distribución de labels.
        """
        counts = {label.value: 0 for label in TradeLabel}
        confidences = {label.value: [] for label in TradeLabel}
        
        for result in labels:
            counts[result.label.value] += 1
            confidences[result.label.value].append(result.confidence)
        
        total = len(labels)
        
        distribution = {}
        for label_value, count in counts.items():
            distribution[label_value] = {
                "count": count,
                "pct": count / total if total > 0 else 0,
                "avg_confidence": (
                    sum(confidences[label_value]) / len(confidences[label_value])
                    if confidences[label_value] else 0
                ),
            }
        
        return distribution
    
    def calculate_quality_metrics(
        self, 
        trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calcula métricas de calidad de las predicciones.
        """
        if not trades:
            return {}
        
        labels = self.batch_label(trades)
        
        pump_count = sum(1 for l in labels if l.label == TradeLabel.PUMP)
        rug_count = sum(1 for l in labels if l.label == TradeLabel.RUG)
        total = len(labels)
        
        winning_pnl = [t["pnl_pct"] for t, l in zip(trades, labels) 
                       if l.label == TradeLabel.PUMP]
        losing_pnl = [t["pnl_pct"] for t, l in zip(trades, labels) 
                      if l.label == TradeLabel.RUG]
        
        avg_win = sum(winning_pnl) / len(winning_pnl) if winning_pnl else 0
        avg_loss = sum(losing_pnl) / len(losing_pnl) if losing_pnl else 0
        
        return {
            "pump_rate": pump_count / total if total > 0 else 0,
            "rug_rate": rug_count / total if total > 0 else 0,
            "neutral_rate": (total - pump_count - rug_count) / total if total > 0 else 0,
            "avg_winning_pnl": avg_win,
            "avg_losing_pnl": avg_loss,
            "win_loss_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            "total_trades": total,
        }
    
    def adjust_thresholds(
        self,
        trades: List[Dict[str, Any]],
        target_pump_rate: float = 0.5,
    ) -> Dict[str, float]:
        """
        Sugiere ajustes a los thresholds basado en datos históricos.
        
        Busca thresholds que produzcan la distribución objetivo.
        """
        if len(trades) < 20:
            return {
                "alpha": self.alpha,
                "beta": self.beta,
                "reason": "Insufficient data for adjustment",
            }
        
        mfes = sorted([t.get("mfe", 0) for t in trades], reverse=True)
        maes = sorted([t.get("mae", 0) for t in trades])
        
        target_idx = int(len(mfes) * target_pump_rate)
        suggested_alpha = mfes[target_idx] if target_idx < len(mfes) else self.alpha
        
        rug_idx = int(len(maes) * (1 - target_pump_rate))
        suggested_beta = abs(maes[rug_idx]) if rug_idx < len(maes) else self.beta
        
        return {
            "alpha": max(0.10, min(suggested_alpha, 0.50)),
            "beta": max(0.15, min(suggested_beta, 0.50)),
            "reason": f"Adjusted to target {target_pump_rate:.0%} pump rate",
            "current_alpha": self.alpha,
            "current_beta": self.beta,
        }
