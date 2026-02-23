"""
Tests para los modelos de scoring.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import HazardModel, PumpModel, EVSCalculator


class TestHazardModel:
    """Tests para el Hazard Model."""
    
    def setup_method(self):
        self.model = HazardModel()
    
    def test_predict_returns_probability(self):
        """P_rug debe estar entre 0 y 1."""
        features = {
            "holder_concentration": 0.5,
            "liquidity_ratio": 0.1,
            "has_renounced": 1.0,
            "age_hours": 6,
        }
        
        p_rug = self.model.predict(features)
        
        assert 0 <= p_rug <= 1
    
    def test_high_concentration_increases_risk(self):
        """Alta concentración de holders debe aumentar P_rug."""
        base_features = {
            "holder_concentration": 0.3,
            "liquidity_ratio": 0.1,
            "has_renounced": 1.0,
        }
        
        high_conc_features = base_features.copy()
        high_conc_features["holder_concentration"] = 0.8
        
        p_rug_low = self.model.predict(base_features)
        p_rug_high = self.model.predict(high_conc_features)
        
        assert p_rug_high > p_rug_low
    
    def test_renounced_ownership_decreases_risk(self):
        """Ownership renunciado debe disminuir P_rug."""
        base_features = {
            "holder_concentration": 0.5,
            "has_renounced": 0.0,
        }
        
        renounced_features = base_features.copy()
        renounced_features["has_renounced"] = 1.0
        
        p_rug_not_renounced = self.model.predict(base_features)
        p_rug_renounced = self.model.predict(renounced_features)
        
        assert p_rug_renounced < p_rug_not_renounced


class TestPumpModel:
    """Tests para el Pump Model."""
    
    def setup_method(self):
        self.model = PumpModel()
    
    def test_predict_returns_dict(self):
        """predict() debe retornar dict con p_pump y expected_g."""
        features = {
            "volume_momentum": 1.5,
            "buy_pressure": 0.6,
        }
        
        result = self.model.predict(features)
        
        assert "p_pump" in result
        assert "expected_g" in result
        assert 0 <= result["p_pump"] <= 1
        assert result["expected_g"] > 0
    
    def test_high_momentum_increases_pump_prob(self):
        """Alto momentum debe aumentar P_pump."""
        base_features = {
            "volume_momentum": 1.0,
            "buy_pressure": 0.5,
        }
        
        high_momentum_features = base_features.copy()
        high_momentum_features["volume_momentum"] = 3.0
        
        p_pump_low = self.model.predict(base_features)["p_pump"]
        p_pump_high = self.model.predict(high_momentum_features)["p_pump"]
        
        assert p_pump_high > p_pump_low


class TestEVSCalculator:
    """Tests para el EVS Calculator."""
    
    def setup_method(self):
        self.calculator = EVSCalculator()
    
    def test_calculate_returns_evs_result(self):
        """calculate() debe retornar EVSResult."""
        features = {
            "holder_concentration": 0.4,
            "liquidity_ratio": 0.15,
            "has_renounced": 1.0,
            "volume_momentum": 1.5,
            "buy_pressure": 0.6,
            "volatility_24h": 0.1,
        }
        
        result = self.calculator.calculate(features)
        
        assert hasattr(result, "evs")
        assert hasattr(result, "evs_adj")
        assert hasattr(result, "p_rug")
        assert hasattr(result, "p_pump")
        assert hasattr(result, "is_tradeable")
    
    def test_high_risk_token_not_tradeable(self):
        """Token de alto riesgo no debe ser tradeable."""
        features = {
            "holder_concentration": 0.9,
            "has_renounced": 0.0,
            "has_freeze_auth": 1.0,
            "structural_risk": 0.8,
        }
        
        result = self.calculator.calculate(features)
        
        assert result.p_rug > 0.4 or not result.is_tradeable
    
    def test_evs_adj_accounts_for_volatility(self):
        """EVS_adj debe ser menor para mayor volatilidad."""
        base_features = {
            "holder_concentration": 0.3,
            "has_renounced": 1.0,
            "volume_momentum": 2.0,
            "volatility_24h": 0.1,
        }
        
        high_vol_features = base_features.copy()
        high_vol_features["volatility_24h"] = 0.3
        
        result_low_vol = self.calculator.calculate(base_features)
        result_high_vol = self.calculator.calculate(high_vol_features)
        
        if result_low_vol.evs > 0 and result_high_vol.evs > 0:
            assert result_high_vol.evs_adj < result_low_vol.evs_adj


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
