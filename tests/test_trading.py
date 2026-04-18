"""
Tests para el módulo de trading.
"""

import pytest
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading import KellyCalculator
from src.calibration import LabelGenerator


@dataclass
class MockEVSResult:
    """Mock de EVSResult para tests."""
    evs: float = 0.05
    evs_adj: float = 0.08
    p_rug: float = 0.25
    p_pump: float = 0.55
    expected_g: float = 0.25
    sigma_token: float = 0.15
    is_tradeable: bool = True
    rejection_reason: str = None


class TestKellyCalculator:
    """Tests para el Kelly Calculator."""
    
    def setup_method(self):
        self.calculator = KellyCalculator(
            gamma=0.25,
            max_position_pct=0.05,
        )
        self.calculator.set_capital(10000)
    
    def test_classic_kelly_positive_expectancy(self):
        """Kelly clásico con expectativa positiva."""
        kelly = self.calculator.calculate_classic_kelly(
            win_prob=0.6,
            win_ratio=1.5,
        )
        
        assert kelly > 0
        assert kelly < 1
    
    def test_classic_kelly_negative_expectancy(self):
        """Kelly clásico con expectativa negativa debe ser 0."""
        kelly = self.calculator.calculate_classic_kelly(
            win_prob=0.3,
            win_ratio=1.0,
        )
        
        assert kelly == 0
    
    def test_position_respects_max(self):
        """Posición no debe exceder máximo."""
        evs_result = MockEVSResult(evs_adj=0.5)
        
        position = self.calculator.calculate_position(evs_result)
        
        assert position.position_pct <= self.calculator.max_position_pct
    
    def test_not_tradeable_returns_zero(self):
        """Token no tradeable debe retornar posición 0."""
        evs_result = MockEVSResult(
            is_tradeable=False,
            rejection_reason="P_rug too high"
        )
        
        position = self.calculator.calculate_position(evs_result)
        
        assert position.position_usd == 0
        assert not position.is_valid
    
    def test_stop_levels_calculated(self):
        """Debe calcular niveles de stop."""
        evs_result = MockEVSResult()
        
        stops = self.calculator.calculate_stop_levels(
            entry_price=0.001,
            evs_result=evs_result,
        )
        
        assert "stop_price" in stops
        assert "take_profit_price" in stops
        assert stops["stop_price"] < 0.001
        assert stops["take_profit_price"] > 0.001


class TestLabelGenerator:
    """Tests para el Label Generator."""
    
    def setup_method(self):
        self.generator = LabelGenerator(
            pump_threshold=0.20,
            default_stop_loss=0.30,
        )
    
    def test_pump_label_on_high_mfe(self):
        """Debe generar label 'pump' cuando MFE > threshold."""
        label = self.generator.generate_label(
            mfe=0.30,
            mae=-0.10,
            pnl=0.20,
        )
        
        assert label == "pump"
    
    def test_rug_label_on_high_mae(self):
        """Debe generar label 'rug' cuando MAE < -threshold."""
        label = self.generator.generate_label(
            mfe=0.05,
            mae=-0.40,
            pnl=-0.35,
        )
        
        assert label == "rug"
    
    def test_neutral_label_on_mild_movement(self):
        """Debe generar label 'neutral' para movimientos leves."""
        label = self.generator.generate_label(
            mfe=0.10,
            mae=-0.15,
            pnl=0.05,
        )
        
        assert label == "neutral"
    
    def test_time_ordering_matters(self):
        """El orden temporal de MFE/MAE debe importar."""
        result_pump_first = self.generator.generate_label_detailed(
            mfe=0.25,
            mae=-0.35,
            pnl=0.10,
            mfe_time=10,
            mae_time=30,
        )
        
        result_rug_first = self.generator.generate_label_detailed(
            mfe=0.25,
            mae=-0.35,
            pnl=-0.20,
            mfe_time=30,
            mae_time=10,
        )
        
        assert result_pump_first.label.value == "pump"
        assert result_rug_first.label.value == "rug"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
