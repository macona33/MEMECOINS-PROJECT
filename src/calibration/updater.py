"""
Model Updater - Recalibración incremental de modelos.
Actualiza Hazard y Pump models con outcomes recientes.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from src.storage import DatabaseManager
from src.models import HazardModel, PumpModel, EVSCalculator
from .labels import LabelGenerator, TradeLabel
from config.settings import SETTINGS


class ModelUpdater:
    """
    Actualizador de modelos con aprendizaje incremental.
    
    Responsabilidades:
    - Recopilar outcomes recientes
    - Preparar datos de entrenamiento
    - Actualizar parámetros de modelos
    - Ajustar thresholds de trading
    """
    
    def __init__(
        self,
        db: DatabaseManager,
        hazard_model: Optional[HazardModel] = None,
        pump_model: Optional[PumpModel] = None,
        evs_calculator: Optional[EVSCalculator] = None,
    ):
        self.db = db
        self.hazard_model = hazard_model or HazardModel()
        self.pump_model = pump_model or PumpModel()
        self.evs_calculator = evs_calculator or EVSCalculator(
            self.hazard_model, self.pump_model
        )
        self.label_generator = LabelGenerator()
        
        self.min_samples = SETTINGS["min_trades_for_recalibration"]
        self.window_days = SETTINGS["recalibration_window_days"]
    
    async def initialize(self) -> None:
        """Inicializa modelos con parámetros guardados."""
        self.hazard_model.set_database(self.db)
        self.pump_model.set_database(self.db)
        
        await self.hazard_model.load_params()
        await self.pump_model.load_params()
        
        logger.info("Model updater initialized")
    
    async def get_training_data(
        self, 
        days: int = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene datos de entrenamiento de trades recientes.
        """
        window = days or self.window_days
        trades = await self.db.get_trade_history(days=window)
        
        training_data = []
        
        for trade in trades:
            features = await self.db.get_features(trade["token_address"])
            
            if not features:
                continue
            
            label = trade.get("label", "neutral")
            
            training_data.append({
                "token_address": trade["token_address"],
                "features": features,
                "label": label,
                "is_rug": label == "rug",
                "is_pump": label == "pump",
                "mfe": trade.get("mfe", 0),
                "mae": trade.get("mae", 0),
                "pnl_pct": trade.get("pnl_pct", 0),
                "actual_g": trade.get("mfe", 0) if label == "pump" else 0,
                "evs_at_entry": trade.get("evs_at_entry", 0),
                "p_rug_at_entry": trade.get("p_rug_at_entry", 0),
                "p_pump_at_entry": trade.get("p_pump_at_entry", 0),
            })
        
        return training_data
    
    async def update_hazard_model(
        self, 
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Actualiza el Hazard Model con outcomes recientes.
        """
        training_data = await self.get_training_data()
        
        if len(training_data) < self.min_samples:
            return {
                "updated": False,
                "reason": f"Insufficient samples: {len(training_data)} < {self.min_samples}",
            }
        
        outcomes = [
            {"features": d["features"], "is_rug": d["is_rug"]}
            for d in training_data
        ]
        
        changes = self.hazard_model.update_params(outcomes, learning_rate)
        
        await self.hazard_model.save_params()
        
        accuracy = self._calculate_hazard_accuracy(training_data)
        
        return {
            "updated": True,
            "samples": len(training_data),
            "param_changes": changes,
            "accuracy": accuracy,
        }
    
    async def update_pump_model(
        self, 
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Actualiza el Pump Model con outcomes recientes.
        """
        training_data = await self.get_training_data()
        
        if len(training_data) < self.min_samples:
            return {
                "updated": False,
                "reason": f"Insufficient samples: {len(training_data)} < {self.min_samples}",
            }
        
        outcomes = [
            {
                "features": d["features"],
                "is_pump": d["is_pump"],
                "actual_g": d["actual_g"],
            }
            for d in training_data
        ]
        
        changes = self.pump_model.update_params(outcomes, learning_rate)
        
        await self.pump_model.save_params()
        
        accuracy = self._calculate_pump_accuracy(training_data)
        
        return {
            "updated": True,
            "samples": len(training_data),
            "param_changes": changes,
            "accuracy": accuracy,
        }
    
    def _calculate_hazard_accuracy(
        self, 
        data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calcula precisión del modelo de hazard."""
        if not data:
            return {}
        
        correct_predictions = 0
        total_rug = 0
        true_positives = 0
        false_positives = 0
        
        for d in data:
            p_rug = self.hazard_model.predict(d["features"])
            predicted_rug = p_rug > 0.5
            actual_rug = d["is_rug"]
            
            if predicted_rug == actual_rug:
                correct_predictions += 1
            
            if actual_rug:
                total_rug += 1
                if predicted_rug:
                    true_positives += 1
            elif predicted_rug:
                false_positives += 1
        
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0 else 0
        )
        recall = true_positives / total_rug if total_rug > 0 else 0
        
        return {
            "accuracy": correct_predictions / len(data),
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            ),
        }
    
    def _calculate_pump_accuracy(
        self, 
        data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calcula precisión del modelo de pump."""
        if not data:
            return {}
        
        g_errors = []
        correct_predictions = 0
        total_pump = 0
        true_positives = 0
        
        for d in data:
            result = self.pump_model.predict(d["features"])
            predicted_pump = result["p_pump"] > 0.5
            actual_pump = d["is_pump"]
            
            if predicted_pump == actual_pump:
                correct_predictions += 1
            
            if actual_pump:
                total_pump += 1
                if predicted_pump:
                    true_positives += 1
                
                g_error = d["actual_g"] - result["expected_g"]
                g_errors.append(g_error)
        
        return {
            "accuracy": correct_predictions / len(data),
            "recall": true_positives / total_pump if total_pump > 0 else 0,
            "mean_g_error": np.mean(g_errors) if g_errors else 0,
            "g_error_std": np.std(g_errors) if g_errors else 0,
        }
    
    async def recalibrate_all(
        self, 
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Recalibra todos los modelos.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "hazard": await self.update_hazard_model(learning_rate),
            "pump": await self.update_pump_model(learning_rate),
        }
        
        training_data = await self.get_training_data()
        if training_data:
            threshold_suggestions = await self.suggest_threshold_adjustments(
                training_data
            )
            results["threshold_suggestions"] = threshold_suggestions
        
        logger.info(f"Recalibration complete: {results}")
        
        return results
    
    async def suggest_threshold_adjustments(
        self, 
        data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sugiere ajustes a thresholds de trading basado en performance.
        """
        if data is None:
            data = await self.get_training_data()
        
        if not data:
            return {}
        
        hit_rate = sum(1 for d in data if d["is_pump"]) / len(data)
        avg_evs = np.mean([d["evs_at_entry"] for d in data if d["evs_at_entry"]])
        
        suggestions = {}
        
        target_hit_rate = SETTINGS["target_hit_rate"]
        current_evs_threshold = SETTINGS["evs_adj_threshold"]
        
        if hit_rate < target_hit_rate - 0.1:
            suggestions["evs_adj_threshold"] = {
                "current": current_evs_threshold,
                "suggested": current_evs_threshold * 1.2,
                "reason": f"Hit rate {hit_rate:.1%} below target {target_hit_rate:.1%}",
            }
        elif hit_rate > target_hit_rate + 0.1:
            suggestions["evs_adj_threshold"] = {
                "current": current_evs_threshold,
                "suggested": current_evs_threshold * 0.9,
                "reason": f"Hit rate {hit_rate:.1%} above target, can be more aggressive",
            }
        
        avg_pnl = np.mean([d["pnl_pct"] for d in data])
        if avg_pnl < 0:
            current_kelly = SETTINGS["kelly_gamma"]
            suggestions["kelly_gamma"] = {
                "current": current_kelly,
                "suggested": current_kelly * 0.8,
                "reason": f"Negative avg PnL {avg_pnl:.1%}, reduce position sizes",
            }
        
        return suggestions
    
    async def evaluate_model_performance(self) -> Dict[str, Any]:
        """
        Evalúa performance de los modelos vs. outcomes reales.
        """
        data = await self.get_training_data()
        
        if not data:
            return {"error": "No training data available"}
        
        hazard_metrics = self._calculate_hazard_accuracy(data)
        pump_metrics = self._calculate_pump_accuracy(data)
        
        calibration_hazard = self._check_calibration(
            [self.hazard_model.predict(d["features"]) for d in data],
            [d["is_rug"] for d in data]
        )
        
        calibration_pump = self._check_calibration(
            [self.pump_model.predict(d["features"])["p_pump"] for d in data],
            [d["is_pump"] for d in data]
        )
        
        return {
            "samples": len(data),
            "hazard_model": {
                **hazard_metrics,
                "calibration": calibration_hazard,
            },
            "pump_model": {
                **pump_metrics,
                "calibration": calibration_pump,
            },
            "overall_hit_rate": sum(1 for d in data if d["is_pump"]) / len(data),
            "overall_avg_pnl": np.mean([d["pnl_pct"] for d in data]),
        }
    
    def _check_calibration(
        self, 
        predictions: List[float], 
        actuals: List[bool],
        n_bins: int = 5
    ) -> Dict[str, Any]:
        """
        Verifica calibración de probabilidades predichas.
        """
        if not predictions:
            return {}
        
        bins = np.linspace(0, 1, n_bins + 1)
        calibration = []
        
        for i in range(n_bins):
            mask = [(bins[i] <= p < bins[i+1]) for p in predictions]
            bin_preds = [p for p, m in zip(predictions, mask) if m]
            bin_actuals = [a for a, m in zip(actuals, mask) if m]
            
            if bin_preds:
                mean_pred = np.mean(bin_preds)
                mean_actual = np.mean(bin_actuals)
                calibration.append({
                    "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    "mean_predicted": mean_pred,
                    "mean_actual": mean_actual,
                    "count": len(bin_preds),
                    "gap": abs(mean_pred - mean_actual),
                })
        
        avg_gap = np.mean([c["gap"] for c in calibration]) if calibration else 0
        
        return {
            "bins": calibration,
            "avg_calibration_gap": avg_gap,
            "well_calibrated": avg_gap < 0.1,
        }
