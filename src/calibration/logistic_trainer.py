"""
Incremental Logistic Regression Trainer v2.0.
Entrena modelos logísticos de forma incremental usando datos de trades.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from config.settings import SETTINGS


@dataclass
class TrainingResult:
    """Resultado de entrenamiento."""
    success: bool
    old_params: Dict[str, float]
    new_params: Dict[str, float]
    n_samples: int
    loss_before: float
    loss_after: float
    improvement: float
    message: str


class IncrementalLogisticRegressor:
    """
    Regresor logístico incremental con regularización L2.
    
    Características:
    - Usa coeficientes actuales como prior (warm start)
    - Regularización L2 para estabilidad
    - Mini-batch updates
    - Fallback a prior si datos insuficientes
    """
    
    def __init__(
        self,
        feature_names: List[str],
        prior_params: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.1,
        l2_lambda: float = 0.01,
        min_samples: int = 30,
    ):
        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.min_samples = min_samples
        
        if prior_params:
            self.params = prior_params.copy()
        else:
            self.params = {name: 0.0 for name in feature_names}
            self.params["intercept"] = 0.0
    
    def _params_to_vector(self) -> np.ndarray:
        """Convierte parámetros dict a vector para optimización."""
        vec = [self.params.get("intercept", 0.0)]
        for name in self.feature_names:
            vec.append(self.params.get(name, 0.0))
        return np.array(vec)
    
    def _vector_to_params(self, vec: np.ndarray) -> Dict[str, float]:
        """Convierte vector a dict de parámetros."""
        params = {"intercept": vec[0]}
        for i, name in enumerate(self.feature_names):
            params[name] = vec[i + 1]
        return params
    
    def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extrae features de un data point."""
        features = []
        for name in self.feature_names:
            val = data.get(name, 0.0)
            if val is None:
                val = 0.0
            features.append(float(val))
        return np.array(features)
    
    def predict_proba(self, features: Dict[str, Any]) -> float:
        """Predice probabilidad para un sample."""
        x = self._extract_features(features)
        z = self.params.get("intercept", 0.0)
        for i, name in enumerate(self.feature_names):
            z += self.params.get(name, 0.0) * x[i]
        return expit(z)
    
    def _loss_function(
        self, 
        theta: np.ndarray, 
        X: np.ndarray, 
        y: np.ndarray,
        prior_theta: np.ndarray,
    ) -> float:
        """
        Función de pérdida: Binary Cross-Entropy + L2 regularization + Prior penalty.
        
        Loss = BCE + λ * ||θ||² + γ * ||θ - θ_prior||²
        """
        z = X @ theta
        z = np.clip(z, -500, 500)
        probs = expit(z)
        
        eps = 1e-10
        probs = np.clip(probs, eps, 1 - eps)
        
        bce = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        l2_penalty = self.l2_lambda * np.sum(theta[1:] ** 2)
        prior_penalty = self.learning_rate * np.sum((theta - prior_theta) ** 2)
        
        return bce + l2_penalty + prior_penalty
    
    def _loss_gradient(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        prior_theta: np.ndarray,
    ) -> np.ndarray:
        """Gradiente de la función de pérdida."""
        z = X @ theta
        z = np.clip(z, -500, 500)
        probs = expit(z)
        
        n = len(y)
        grad_bce = X.T @ (probs - y) / n
        
        grad_l2 = np.zeros_like(theta)
        grad_l2[1:] = 2 * self.l2_lambda * theta[1:]
        
        grad_prior = 2 * self.learning_rate * (theta - prior_theta)
        
        return grad_bce + grad_l2 + grad_prior
    
    def fit(
        self,
        training_data: List[Dict[str, Any]],
        label_key: str = "is_rug",
    ) -> TrainingResult:
        """
        Entrena el modelo con nuevos datos.
        
        Args:
            training_data: Lista de dicts con features y labels
            label_key: Nombre del campo que contiene el label (0/1)
            
        Returns:
            TrainingResult con métricas del entrenamiento
        """
        old_params = self.params.copy()
        
        if len(training_data) < self.min_samples:
            return TrainingResult(
                success=False,
                old_params=old_params,
                new_params=old_params,
                n_samples=len(training_data),
                loss_before=0,
                loss_after=0,
                improvement=0,
                message=f"Insufficient data: {len(training_data)} < {self.min_samples}",
            )
        
        X_list = []
        y_list = []
        
        for data in training_data:
            features = self._extract_features(data)
            label = data.get(label_key, 0)
            if label is None:
                continue
            X_list.append(np.concatenate([[1.0], features]))
            y_list.append(float(label))
        
        if len(y_list) < self.min_samples:
            return TrainingResult(
                success=False,
                old_params=old_params,
                new_params=old_params,
                n_samples=len(y_list),
                loss_before=0,
                loss_after=0,
                improvement=0,
                message=f"Too few valid samples: {len(y_list)}",
            )
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        prior_theta = self._params_to_vector()
        initial_theta = prior_theta.copy()
        
        loss_before = self._loss_function(initial_theta, X, y, prior_theta)
        
        result = minimize(
            fun=self._loss_function,
            x0=initial_theta,
            args=(X, y, prior_theta),
            method="L-BFGS-B",
            jac=self._loss_gradient,
            options={"maxiter": 100, "disp": False},
        )
        
        if result.success:
            new_theta = result.x
            new_theta = prior_theta + self.learning_rate * (new_theta - prior_theta)
            loss_after = self._loss_function(new_theta, X, y, prior_theta)
            
            self.params = self._vector_to_params(new_theta)
            
            improvement = (loss_before - loss_after) / loss_before if loss_before > 0 else 0
            
            return TrainingResult(
                success=True,
                old_params=old_params,
                new_params=self.params.copy(),
                n_samples=len(y),
                loss_before=loss_before,
                loss_after=loss_after,
                improvement=improvement,
                message=f"Training successful, loss improved {improvement:.1%}",
            )
        else:
            return TrainingResult(
                success=False,
                old_params=old_params,
                new_params=old_params,
                n_samples=len(y),
                loss_before=loss_before,
                loss_after=loss_before,
                improvement=0,
                message=f"Optimization failed: {result.message}",
            )
    
    def get_params(self) -> Dict[str, float]:
        """Retorna copia de parámetros actuales."""
        return self.params.copy()
    
    def set_params(self, params: Dict[str, float]) -> None:
        """Establece parámetros."""
        self.params = params.copy()
    
    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        label_key: str = "is_rug",
    ) -> Dict[str, float]:
        """
        Evalúa el modelo en datos de test.
        
        Returns:
            Dict con métricas: accuracy, precision, recall, brier_score
        """
        if not test_data:
            return {}
        
        y_true = []
        y_pred = []
        y_prob = []
        
        for data in test_data:
            label = data.get(label_key)
            if label is None:
                continue
            
            prob = self.predict_proba(data)
            pred = 1 if prob >= 0.5 else 0
            
            y_true.append(float(label))
            y_pred.append(pred)
            y_prob.append(prob)
        
        if not y_true:
            return {}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        accuracy = np.mean(y_true == y_pred)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        brier_score = np.mean((y_prob - y_true) ** 2)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "brier_score": brier_score,
            "n_samples": len(y_true),
            "n_positive": int(np.sum(y_true)),
            "n_negative": int(np.sum(1 - y_true)),
        }


class HazardModelTrainer:
    """
    Trainer específico para el Hazard Model (P_rug).
    """
    
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
    
    DEFAULT_PRIOR = {
        "intercept": -1.5,
        "holder_concentration": 2.0,
        "top_holder_pct": 1.5,
        "liquidity_ratio": -3.0,
        "age_hours": -0.1,
        "has_renounced": -1.0,
        "has_verified": -0.5,
        "has_freeze_auth": 0.8,
        "has_mint_auth": 0.6,
        "volume_liquidity_ratio": -0.3,
        "structural_risk": 1.5,
    }
    
    def __init__(self, prior_params: Optional[Dict[str, float]] = None):
        prior = prior_params or self.DEFAULT_PRIOR
        self.regressor = IncrementalLogisticRegressor(
            feature_names=self.FEATURE_NAMES,
            prior_params=prior,
            min_samples=SETTINGS.get("min_trades_for_recalibration", 30),
        )
    
    def prepare_training_data(
        self, 
        trade_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepara datos de trade_features para entrenamiento.
        Label: is_rug = 1 si stop_executed o label == 'rug'
        """
        training_data = []
        
        for tf in trade_features:
            is_rug = (
                tf.get("stop_executed", 0) == 1 or 
                tf.get("label", "").lower() == "rug"
            )
            
            data = {name: tf.get(name, 0) for name in self.FEATURE_NAMES}
            data["is_rug"] = 1 if is_rug else 0
            training_data.append(data)
        
        return training_data
    
    def train(self, trade_features: List[Dict[str, Any]]) -> TrainingResult:
        """Entrena con datos de trade_features."""
        training_data = self.prepare_training_data(trade_features)
        return self.regressor.fit(training_data, label_key="is_rug")
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predice P_rug para features dadas."""
        return self.regressor.predict_proba(features)
    
    def get_params(self) -> Dict[str, float]:
        return self.regressor.get_params()
    
    def set_params(self, params: Dict[str, float]) -> None:
        self.regressor.set_params(params)


class PumpModelTrainer:
    """
    Trainer específico para el Pump Model (P_pump).
    """
    
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
    
    DEFAULT_PRIOR = {
        "intercept": -0.5,
        "volume_momentum": 0.8,
        "price_momentum": 0.6,
        "tx_momentum": 0.4,
        "buy_pressure": 1.2,
        "liquidity_quality": 0.5,
        "opportunity_score": 1.0,
        "log_liquidity": 0.1,
        "volatility_24h": 0.3,
        "momentum_score": 0.5,
    }
    
    def __init__(self, prior_params: Optional[Dict[str, float]] = None):
        prior = prior_params or self.DEFAULT_PRIOR
        self.regressor = IncrementalLogisticRegressor(
            feature_names=self.FEATURE_NAMES,
            prior_params=prior,
            min_samples=SETTINGS.get("min_trades_for_recalibration", 30),
        )
    
    def prepare_training_data(
        self, 
        trade_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepara datos para entrenamiento.
        Label: is_pump = 1 si MFE >= take_profit antes del stop
        """
        training_data = []
        
        for tf in trade_features:
            is_pump = tf.get("label", "").lower() == "pump"
            
            data = {name: tf.get(name, 0) for name in self.FEATURE_NAMES}
            data["is_pump"] = 1 if is_pump else 0
            training_data.append(data)
        
        return training_data
    
    def train(self, trade_features: List[Dict[str, Any]]) -> TrainingResult:
        """Entrena con datos de trade_features."""
        training_data = self.prepare_training_data(trade_features)
        return self.regressor.fit(training_data, label_key="is_pump")
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predice P_pump para features dadas."""
        return self.regressor.predict_proba(features)
    
    def get_params(self) -> Dict[str, float]:
        return self.regressor.get_params()
    
    def set_params(self, params: Dict[str, float]) -> None:
        self.regressor.set_params(params)
