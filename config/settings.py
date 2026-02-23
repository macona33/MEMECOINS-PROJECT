"""
Configuración global del sistema de paper trading v2.0.
Sistema paramétrico adaptativo con edge estable.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TIMESERIES_DIR = DATA_DIR / "timeseries"

SETTINGS = {
    # ============== FILTROS DE ENTRADA ==============
    "min_liquidity_usd": 10_000,
    "require_renounced": True,
    "require_verified": False,
    "max_holder_concentration": 0.50,
    "min_token_age_minutes": 5,
    "max_token_age_hours": 24,
    
    # ============== THRESHOLDS DE SCORING ==============
    "evs_adj_threshold": 0.05,
    "max_p_rug": 0.40,
    "min_p_pump": 0.20,
    
    # ============== TRADING v2.0 ==============
    "initial_capital": 10_000,
    "kelly_gamma": 0.10,              # v2.0: Reducido de 0.25 a 0.10
    "max_position_pct": 0.03,         # v2.0: Reducido de 0.05 a 0.03
    "early_stage_max_position": 0.03, # v2.0: Máximo mientras < 100 trades
    "mature_stage_max_position": 0.05,# v2.0: Máximo después de >= 100 trades
    "base_stop_loss": 0.15,
    "take_profit_target": 0.30,
    "max_hold_hours": 24,
    "max_concurrent_trades": 10,
    
    # ============== v2.0: CAPS DE KELLY ==============
    "min_sigma_cap": 0.05,            # Si sigma < esto, aplicar cap conservador
    "low_sigma_max_kelly": 0.02,      # Max Kelly si sigma muy baja
    "min_trades_for_full_kelly": 100, # Trades mínimos para Kelly completo
    
    # ============== COSTES ESTIMADOS ==============
    "slippage_pct": 0.02,
    "fees_pct": 0.01,
    "rug_loss_pct": 0.95,
    
    # ============== LABELS PATH-DEPENDENT v2.0 ==============
    "pump_threshold_alpha": 0.20,
    # v2.0: rug_threshold_beta eliminado - ahora usa stop_loss real
    
    # ============== POLLING ==============
    "scan_interval_seconds": 120,
    "price_poll_seconds": 30,
    "dexscreener_rate_limit": 30,
    
    # ============== RECALIBRACIÓN v2.0 ==============
    "min_trades_for_recalibration": 30,  # v2.0: Reducido de 50 a 30
    "recalibration_window_days": 7,
    "recalibration_interval_hours": 24,  # v2.0: Cada 24 horas
    "new_trades_trigger": 30,            # v2.0: O cada N trades nuevos
    
    # ============== v2.0: DETECTOR DE RÉGIMEN ==============
    "regime_check_hours": 2,             # Ventana para calcular régimen
    "regime_low_tokens_threshold": 20,   # < 20 tokens = LOW
    "regime_high_tokens_threshold": 100, # > 100 tokens = HIGH
    "regime_low_volume_pct": 0.30,       # < 30% con volumen = LOW
    "regime_high_volume_pct": 0.70,      # > 70% con volumen = HIGH
    "regime_volume_threshold": 5000,     # Volumen mínimo para contar
    
    # ============== v2.0: AJUSTES POR RÉGIMEN ==============
    "regime_adjustments": {
        "LOW_ACTIVITY": {
            "evs_threshold_mult": 1.20,      # +20% threshold
            "max_position_mult": 0.50,       # -50% position
            "max_concurrent_mult": 0.50,     # -50% concurrent
        },
        "NORMAL": {
            "evs_threshold_mult": 1.00,
            "max_position_mult": 1.00,
            "max_concurrent_mult": 1.00,
        },
        "HIGH_ACTIVITY": {
            "evs_threshold_mult": 0.90,      # -10% threshold (más oportunidades)
            "max_position_mult": 1.00,
            "max_concurrent_mult": 1.20,     # +20% concurrent
        },
    },
    
    # ============== v2.0: CONTROL DE DRAWDOWN ==============
    "drawdown_level_1": 0.10,         # 10% drawdown
    "drawdown_gamma_mult_1": 0.75,    # Reducir gamma 25%
    "drawdown_level_2": 0.20,         # 20% drawdown
    "drawdown_gamma_mult_2": 0.50,    # Reducir gamma 50%
    "drawdown_freeze_level": 0.30,    # 30% drawdown = freeze
    "drawdown_freeze_hours": 2,       # Freeze por 2 horas
    "consecutive_loss_threshold": 3,  # 3 pérdidas consecutivas
    "consecutive_loss_size_mult": 0.50, # Reducir tamaño 50%
    
    # ============== v2.0: G DINÁMICO ==============
    "g_buckets": [
        {"id": "low", "min": 0.05, "max": 0.10},
        {"id": "medium", "min": 0.10, "max": 0.15},
        {"id": "high", "min": 0.15, "max": 0.20},
        {"id": "very_high", "min": 0.20, "max": 1.00},
    ],
    "g_fallback": 0.25,               # G conservador si no hay datos
    "g_min_samples": 5,               # Mínimo de samples por bucket
    
    # ============== CRITERIOS DE ÉXITO ==============
    "target_hit_rate": 0.60,
    "max_drawdown_tolerance": 0.40,
    "min_sharpe_ratio": 0.5,
}

DATABASE_PATH = DATA_DIR / "tokens.db"
