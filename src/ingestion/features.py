"""
Extractor de features para scoring de tokens.
Calcula variables estructurales y dinámicas para los modelos.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

from src.storage import DatabaseManager, TimeseriesManager
from config.settings import SETTINGS


class FeatureExtractor:
    """Extrae y calcula features para el scoring de tokens."""
    
    def __init__(
        self, 
        db: DatabaseManager,
        timeseries: Optional[TimeseriesManager] = None
    ):
        self.db = db
        self.timeseries = timeseries or TimeseriesManager()
    
    def extract_static_features(self, token: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrae features estáticas (no cambian con el tiempo).
        """
        features = {}
        
        features["has_renounced"] = 1.0 if token.get("ownership_renounced") else 0.0
        features["has_verified"] = 1.0 if token.get("contract_verified") else 0.0
        features["has_freeze_auth"] = 1.0 if token.get("has_freeze_authority") else 0.0
        features["has_mint_auth"] = 1.0 if token.get("has_mint_authority") else 0.0
        
        features["holder_concentration"] = token.get("top_10_holders_pct", 0.5) or 0.5
        features["top_holder_pct"] = token.get("top_holder_pct", 0.2) or 0.2
        features["holder_count"] = float(token.get("holder_count", 0) or 0)
        
        liquidity = token.get("liquidity_usd", 0) or 0
        features["liquidity_usd"] = liquidity
        features["log_liquidity"] = np.log1p(liquidity)
        
        market_cap = token.get("market_cap", 0) or 0
        features["market_cap"] = market_cap
        features["log_market_cap"] = np.log1p(market_cap)
        
        if market_cap > 0:
            features["liquidity_ratio"] = liquidity / market_cap
        else:
            features["liquidity_ratio"] = 0.0
        
        return features
    
    def extract_dynamic_features(self, token: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrae features dinámicas (cambian con el tiempo).
        """
        features = {}
        
        age_hours = token.get("age_hours", 24) or 24
        features["age_hours"] = age_hours
        features["log_age_hours"] = np.log1p(age_hours)
        features["is_very_new"] = 1.0 if age_hours < 1 else 0.0
        
        volume_24h = token.get("volume_24h", 0) or 0
        volume_1h = token.get("volume_1h", 0) or 0
        features["volume_24h"] = volume_24h
        features["volume_1h"] = volume_1h
        features["log_volume_24h"] = np.log1p(volume_24h)
        
        if volume_24h > 0:
            features["volume_momentum"] = (volume_1h * 24) / volume_24h
        else:
            features["volume_momentum"] = 1.0
        
        liquidity = token.get("liquidity_usd", 1) or 1
        features["volume_liquidity_ratio"] = volume_24h / liquidity
        
        price_change_1h = token.get("price_change_1h", 0) or 0
        price_change_24h = token.get("price_change_24h", 0) or 0
        features["price_change_1h"] = price_change_1h / 100
        features["price_change_24h"] = price_change_24h / 100
        
        features["price_momentum"] = (1 + price_change_1h/100) / max(1 + price_change_24h/100, 0.01)
        
        tx_count_1h = token.get("tx_count_1h", 0) or 0
        tx_count_24h = token.get("tx_count_24h", 0) or 0
        features["tx_count_1h"] = float(tx_count_1h)
        features["tx_count_24h"] = float(tx_count_24h)
        
        if tx_count_24h > 0:
            features["tx_momentum"] = (tx_count_1h * 24) / tx_count_24h
        else:
            features["tx_momentum"] = 1.0
        
        buy_sell_ratio = token.get("buy_sell_ratio", 1) or 1
        features["buy_sell_ratio"] = min(buy_sell_ratio, 10)
        features["buy_pressure"] = buy_sell_ratio / (1 + buy_sell_ratio)
        
        return features
    
    async def extract_volatility_features(self, token_address: str) -> Dict[str, float]:
        """
        Calcula features de volatilidad usando datos históricos.
        """
        features = {
            "volatility_1h": 0.05,
            "volatility_24h": 0.10,
            "max_drawdown_24h": 0.0,
            "max_pump_24h": 0.0,
        }
        
        vol_24h = await self.timeseries.calculate_volatility(token_address, window_hours=24)
        if vol_24h is not None:
            features["volatility_24h"] = vol_24h
        
        vol_1h = await self.timeseries.calculate_volatility(token_address, window_hours=1)
        if vol_1h is not None:
            features["volatility_1h"] = vol_1h
        
        from datetime import timedelta
        df = await self.timeseries.get_token_prices(
            token_address,
            start_date=datetime.now() - timedelta(hours=24)
        )
        
        if not df.empty and len(df) > 1:
            prices = df["price_usd"].values
            
            cummax = np.maximum.accumulate(prices)
            drawdowns = (prices - cummax) / cummax
            features["max_drawdown_24h"] = abs(drawdowns.min())
            
            cummin = np.minimum.accumulate(prices)
            pumps = (prices - cummin) / np.maximum(cummin, 1e-10)
            features["max_pump_24h"] = pumps.max()
        
        return features
    
    def calculate_risk_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Calcula features de riesgo compuestas.
        """
        risk_features = {}
        
        risk_score = 0.0
        
        if features.get("has_freeze_auth", 0) > 0:
            risk_score += 0.2
        if features.get("has_mint_auth", 0) > 0:
            risk_score += 0.2
        if features.get("has_renounced", 0) == 0:
            risk_score += 0.3
        
        holder_conc = features.get("holder_concentration", 0.5)
        risk_score += holder_conc * 0.3
        
        liq_ratio = features.get("liquidity_ratio", 0)
        if liq_ratio < 0.05:
            risk_score += 0.2
        
        risk_features["structural_risk"] = min(risk_score, 1.0)
        
        dyn_risk = 0.0
        
        vol = features.get("volatility_24h", 0.1)
        dyn_risk += min(vol / 0.5, 1.0) * 0.3
        
        vol_liq = features.get("volume_liquidity_ratio", 0)
        if vol_liq > 5:
            dyn_risk += 0.2
        
        buy_sell = features.get("buy_sell_ratio", 1)
        if buy_sell > 5 or buy_sell < 0.2:
            dyn_risk += 0.2
        
        risk_features["dynamic_risk"] = min(dyn_risk, 1.0)
        
        risk_features["total_risk"] = (
            risk_features["structural_risk"] * 0.6 + 
            risk_features["dynamic_risk"] * 0.4
        )
        
        return risk_features
    
    def calculate_opportunity_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Calcula features de oportunidad.
        """
        opp_features = {}
        
        momentum_score = 0.0
        
        vol_momentum = features.get("volume_momentum", 1)
        if vol_momentum > 1:
            momentum_score += min((vol_momentum - 1) / 3, 0.3)
        
        price_momentum = features.get("price_momentum", 1)
        if price_momentum > 1:
            momentum_score += min((price_momentum - 1) / 2, 0.3)
        
        tx_momentum = features.get("tx_momentum", 1)
        if tx_momentum > 1:
            momentum_score += min((tx_momentum - 1) / 3, 0.2)
        
        buy_pressure = features.get("buy_pressure", 0.5)
        momentum_score += (buy_pressure - 0.5) * 0.4
        
        opp_features["momentum_score"] = max(0, min(momentum_score, 1.0))
        
        liq_score = 0.0
        
        log_liq = features.get("log_liquidity", 0)
        liq_score += min(log_liq / 15, 0.5)
        
        liq_ratio = features.get("liquidity_ratio", 0)
        if liq_ratio > 0.1:
            liq_score += 0.3
        
        vol_liq = features.get("volume_liquidity_ratio", 0)
        if 0.5 < vol_liq < 3:
            liq_score += 0.2
        
        opp_features["liquidity_score"] = min(liq_score, 1.0)
        
        opp_features["opportunity_score"] = (
            opp_features["momentum_score"] * 0.6 +
            opp_features["liquidity_score"] * 0.4
        )
        
        return opp_features
    
    async def extract_all_features(self, token: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrae todas las features para un token.
        """
        all_features = {}
        
        static = self.extract_static_features(token)
        all_features.update(static)
        
        dynamic = self.extract_dynamic_features(token)
        all_features.update(dynamic)
        
        address = token.get("address")
        if address:
            volatility = await self.extract_volatility_features(address)
            all_features.update(volatility)
        
        risk = self.calculate_risk_features(all_features)
        all_features.update(risk)
        
        opportunity = self.calculate_opportunity_features(all_features)
        all_features.update(opportunity)
        
        return all_features
    
    async def extract_and_save(self, token: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrae features y las guarda en la base de datos.
        """
        features = await self.extract_all_features(token)
        
        address = token.get("address")
        if address:
            db_features = {
                "holder_concentration": features.get("holder_concentration"),
                "top_10_holders_pct": features.get("holder_concentration"),
                "liquidity_depth": features.get("liquidity_ratio"),
                "volume_24h": features.get("volume_24h"),
                "volume_1h": features.get("volume_1h"),
                "price_change_1h": features.get("price_change_1h"),
                "price_change_24h": features.get("price_change_24h"),
                "price_volatility": features.get("volatility_24h"),
                "age_hours": features.get("age_hours"),
                "tx_count_1h": features.get("tx_count_1h"),
                "buy_sell_ratio": features.get("buy_sell_ratio"),
            }
            await self.db.insert_features(address, db_features)
        
        return features
    
    async def batch_extract(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Extrae features para múltiples tokens.
        """
        results = []
        for token in tokens:
            try:
                features = await self.extract_and_save(token)
                results.append(features)
            except Exception as e:
                logger.error(f"Error extracting features for {token.get('symbol')}: {e}")
                results.append({})
        
        return results
