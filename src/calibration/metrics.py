"""
Metrics Tracker v2.0 - Seguimiento de métricas de rendimiento y calibración.
v2.0: Añade métricas de calibración (Brier score, calibration error, EV gap).
"""

import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from src.storage import DatabaseManager
from config.settings import SETTINGS


class MetricsTracker:
    """
    Tracker de métricas de rendimiento v2.0.
    
    Métricas principales:
    - Hit Rate: % de trades exitosos (pump)
    - EVS_adj promedio
    - Maximum Drawdown
    - Sharpe Ratio estimado
    - Total PnL
    
    v2.0 Métricas de calibración:
    - Brier Score para P_rug y P_pump
    - Calibration Error por decil
    - EV Gap (real vs estimado)
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self._running_pnl: List[float] = []
        self._equity_curve: List[float] = [SETTINGS["initial_capital"]]
    
    async def calculate_daily_metrics(
        self, 
        target_date: date = None
    ) -> Dict[str, Any]:
        """
        Calcula métricas para un día específico.
        """
        if target_date is None:
            target_date = date.today()
        
        trades = await self.db.get_trade_history(days=1)
        
        trades_today = [
            t for t in trades
            if t.get("exit_time") and 
            datetime.fromisoformat(t["exit_time"]).date() == target_date
        ]
        
        if not trades_today:
            return {
                "date": target_date.isoformat(),
                "total_trades": 0,
                "message": "No trades closed today",
            }
        
        total = len(trades_today)
        
        pump_count = sum(1 for t in trades_today if t.get("label") == "pump")
        rug_count = sum(1 for t in trades_today if t.get("label") == "rug")
        neutral_count = total - pump_count - rug_count
        
        hit_rate = pump_count / total if total > 0 else 0
        
        winning_trades = sum(1 for t in trades_today if (t.get("pnl_pct") or 0) > 0)
        
        evs_values = [t.get("evs_at_entry", 0) for t in trades_today if t.get("evs_at_entry")]
        avg_evs = np.mean(evs_values) if evs_values else 0
        
        pnl_values = [t.get("pnl_pct", 0) for t in trades_today]
        total_pnl_pct = sum(pnl_values)
        
        pnl_usd_values = [t.get("pnl_usd", 0) for t in trades_today]
        total_pnl_usd = sum(pnl_usd_values)
        
        durations = [t.get("duration_minutes", 0) for t in trades_today if t.get("duration_minutes")]
        avg_duration = np.mean(durations) if durations else 0
        
        metrics = {
            "date": target_date.isoformat(),
            "total_trades": total,
            "winning_trades": winning_trades,
            "pump_count": pump_count,
            "rug_count": rug_count,
            "neutral_count": neutral_count,
            "hit_rate": hit_rate,
            "avg_evs_adj": avg_evs,
            "total_pnl_pct": total_pnl_pct,
            "total_pnl_usd": total_pnl_usd,
            "avg_pnl_pct": np.mean(pnl_values) if pnl_values else 0,
            "avg_hold_duration": avg_duration,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
        }
        
        await self.db.update_daily_metrics(target_date.isoformat(), metrics)
        
        return metrics
    
    async def calculate_period_metrics(
        self, 
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calcula métricas para un período.
        """
        trades = await self.db.get_trade_history(days=days)
        
        if not trades:
            return {
                "period_days": days,
                "total_trades": 0,
                "message": "No trades in period",
            }
        
        total = len(trades)
        pump_count = sum(1 for t in trades if t.get("label") == "pump")
        rug_count = sum(1 for t in trades if t.get("label") == "rug")
        
        hit_rate = pump_count / total if total > 0 else 0
        
        pnl_values = [(t.get("pnl_pct") or 0) for t in trades]
        pnl_usd_values = [(t.get("pnl_usd") or 0) for t in trades]
        
        max_dd = self._calculate_max_drawdown(pnl_values)
        
        sharpe = self._calculate_sharpe(pnl_values)
        
        winning = [p for p in pnl_values if p > 0]
        losing = [p for p in pnl_values if p < 0]
        
        avg_win = np.mean(winning) if winning else 0
        avg_loss = np.mean(losing) if losing else 0
        profit_factor = abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else 0
        
        evs_values = [t.get("evs_at_entry", 0) for t in trades if t.get("evs_at_entry")]
        
        return {
            "period_days": days,
            "total_trades": total,
            "trades_per_day": total / days,
            "pump_count": pump_count,
            "rug_count": rug_count,
            "hit_rate": hit_rate,
            "win_rate": len(winning) / total if total > 0 else 0,
            "total_pnl_pct": sum(pnl_values),
            "total_pnl_usd": sum(pnl_usd_values),
            "avg_pnl_pct": np.mean(pnl_values) if pnl_values else 0,
            "pnl_std": np.std(pnl_values) if pnl_values else 0,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_evs_at_entry": np.mean(evs_values) if evs_values else 0,
            "best_trade": max(pnl_values) if pnl_values else 0,
            "worst_trade": min(pnl_values) if pnl_values else 0,
        }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calcula maximum drawdown de una serie de returns.
        """
        if not returns:
            return 0.0
        returns = [float(r) if r is not None else 0.0 for r in returns]
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    def _calculate_sharpe(
        self, 
        returns: List[float], 
        risk_free_rate: float = 0.0,
        periods_per_year: float = 365
    ) -> float:
        """
        Calcula Sharpe ratio anualizado.
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / periods_per_year
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        daily_sharpe = mean_return / std_return
        annualized_sharpe = daily_sharpe * np.sqrt(periods_per_year)
        
        return float(annualized_sharpe)
    
    async def check_success_criteria(self) -> Dict[str, Any]:
        """
        Verifica si se cumplen los criterios de éxito para trading real.
        """
        metrics = await self.calculate_period_metrics(days=30)
        
        if metrics.get("total_trades", 0) < SETTINGS["min_trades_for_recalibration"]:
            return {
                "ready": False,
                "reason": f"Insufficient trades: {metrics.get('total_trades', 0)} < {SETTINGS['min_trades_for_recalibration']}",
                "metrics": metrics,
            }
        
        criteria = {
            "hit_rate": {
                "current": metrics.get("hit_rate", 0),
                "target": SETTINGS["target_hit_rate"],
                "passed": metrics.get("hit_rate", 0) >= SETTINGS["target_hit_rate"],
            },
            "max_drawdown": {
                "current": metrics.get("max_drawdown", 1),
                "target": SETTINGS["max_drawdown_tolerance"],
                "passed": metrics.get("max_drawdown", 1) <= SETTINGS["max_drawdown_tolerance"],
            },
            "sharpe_ratio": {
                "current": metrics.get("sharpe_ratio", 0),
                "target": SETTINGS["min_sharpe_ratio"],
                "passed": metrics.get("sharpe_ratio", 0) >= SETTINGS["min_sharpe_ratio"],
            },
            "positive_pnl": {
                "current": metrics.get("total_pnl_pct", 0),
                "target": 0,
                "passed": metrics.get("total_pnl_pct", 0) > 0,
            },
        }
        
        all_passed = all(c["passed"] for c in criteria.values())
        
        return {
            "ready": all_passed,
            "criteria": criteria,
            "metrics": metrics,
            "recommendation": (
                "System ready for live trading consideration"
                if all_passed
                else "Continue paper trading and model refinement"
            ),
        }
    
    async def generate_report(self, days: int = 7) -> str:
        """
        Genera reporte de métricas en texto.
        """
        metrics = await self.calculate_period_metrics(days)
        success = await self.check_success_criteria()
        
        lines = [
            "=" * 50,
            f"PAPER TRADING REPORT - Last {days} days",
            "=" * 50,
            "",
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Trades/Day: {metrics.get('trades_per_day', 0):.1f}",
            "",
            "--- Performance ---",
            f"Hit Rate: {metrics.get('hit_rate', 0):.1%}",
            f"Win Rate: {metrics.get('win_rate', 0):.1%}",
            f"Total PnL: {metrics.get('total_pnl_pct', 0):.1%} (${metrics.get('total_pnl_usd', 0):.2f})",
            f"Avg PnL: {metrics.get('avg_pnl_pct', 0):.2%}",
            "",
            "--- Risk ---",
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.1%}",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            "",
            "--- Trade Breakdown ---",
            f"Pumps: {metrics.get('pump_count', 0)}",
            f"Rugs: {metrics.get('rug_count', 0)}",
            f"Avg Win: {metrics.get('avg_win', 0):.1%}",
            f"Avg Loss: {metrics.get('avg_loss', 0):.1%}",
            "",
            "--- Success Criteria ---",
        ]
        
        for name, criterion in success.get("criteria", {}).items():
            status = "PASS" if criterion["passed"] else "FAIL"
            lines.append(
                f"  {name}: {status} "
                f"(current: {criterion['current']:.2f}, target: {criterion['target']:.2f})"
            )
        
        lines.extend([
            "",
            f"Recommendation: {success.get('recommendation', 'N/A')}",
            "=" * 50,
        ])
        
        return "\n".join(lines)
    
    async def get_equity_curve(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Genera curva de equity para el período.
        """
        daily_metrics = await self.db.get_daily_metrics(days=days)
        
        equity = SETTINGS["initial_capital"]
        curve = []
        
        for day_metrics in sorted(daily_metrics, key=lambda x: x["date"]):
            pnl = day_metrics.get("total_pnl_usd", 0) or 0
            equity += pnl
            
            curve.append({
                "date": day_metrics["date"],
                "equity": equity,
                "daily_pnl": pnl,
                "daily_return": pnl / (equity - pnl) if (equity - pnl) > 0 else 0,
            })
        
        return curve

    async def get_equity_curve_from_trades(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Genera curva de equity desde historial de trades (más fiable si daily_metrics está vacío).
        """
        from datetime import datetime, timedelta

        trades = await self.db.get_trade_history(days=days)
        closed = [
            t for t in trades
            if t.get("exit_time") and t.get("pnl_usd") is not None
        ]
        closed.sort(key=lambda x: str(x.get("exit_time", "")))
        equity = float(SETTINGS["initial_capital"])
        curve = []
        if closed:
            first_exit = str(closed[0].get("exit_time", ""))[:10]
            try:
                first_dt = datetime.strptime(first_exit, "%Y-%m-%d")
                start_dt = first_dt - timedelta(days=1)
                curve.append({"date": start_dt.strftime("%Y-%m-%d"), "equity": equity})
            except (ValueError, TypeError):
                curve.append({"date": first_exit, "equity": equity})
        for t in closed:
            equity += float(t.get("pnl_usd", 0))
            exit_time = t.get("exit_time", "")
            date_str = exit_time[:10] if exit_time else None
            if date_str:
                curve.append({"date": date_str, "equity": equity})
        if not curve:
            curve.append({"date": datetime.now().strftime("%Y-%m-%d"), "equity": equity})
        return curve
    
    async def export_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Exporta todas las métricas para análisis externo.
        """
        period_metrics = await self.calculate_period_metrics(days)
        success_criteria = await self.check_success_criteria()
        equity_curve = await self.get_equity_curve(days)
        calibration = await self.calculate_calibration_metrics(days)
        
        return {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "summary": period_metrics,
            "success_criteria": success_criteria,
            "equity_curve": equity_curve,
            "calibration": calibration,
            "settings": {
                "initial_capital": SETTINGS["initial_capital"],
                "target_hit_rate": SETTINGS["target_hit_rate"],
                "max_drawdown_tolerance": SETTINGS["max_drawdown_tolerance"],
            },
        }
    
    # ============== v2.0: MÉTRICAS DE CALIBRACIÓN ==============
    
    async def calculate_calibration_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        v2.0: Calcula métricas de calibración de los modelos.
        
        Métricas:
        - Brier Score: Mide precisión de probabilidades
        - Calibration Error: Error por decil de probabilidad
        - EV Gap: Diferencia entre EV estimado y real
        """
        trade_features = await self.db.get_training_dataset(days)
        
        if not trade_features or len(trade_features) < 10:
            return {
                "status": "insufficient_data",
                "n_samples": len(trade_features) if trade_features else 0,
                "message": "Need at least 10 trades for calibration metrics",
            }
        
        p_rug_calibration = self._calculate_probability_calibration(
            trade_features, 
            prob_key="p_rug_predicted",
            outcome_key="label",
            positive_outcome="rug",
        )
        
        p_pump_calibration = self._calculate_probability_calibration(
            trade_features,
            prob_key="p_pump_predicted", 
            outcome_key="label",
            positive_outcome="pump",
        )
        
        ev_gap = self._calculate_ev_gap(trade_features)
        
        return {
            "status": "ok",
            "n_samples": len(trade_features),
            "p_rug_calibration": p_rug_calibration,
            "p_pump_calibration": p_pump_calibration,
            "ev_gap": ev_gap,
        }
    
    def _calculate_probability_calibration(
        self,
        data: List[Dict[str, Any]],
        prob_key: str,
        outcome_key: str,
        positive_outcome: str,
    ) -> Dict[str, Any]:
        """
        Calcula métricas de calibración para una probabilidad.
        
        Returns:
            Dict con brier_score, calibration_error, y calibration_by_decile
        """
        valid_data = [
            d for d in data 
            if d.get(prob_key) is not None and d.get(outcome_key) is not None
        ]
        
        if not valid_data:
            return {"error": "No valid data"}
        
        probs = np.array([d[prob_key] for d in valid_data])
        outcomes = np.array([1.0 if d[outcome_key] == positive_outcome else 0.0 for d in valid_data])
        
        brier_score = np.mean((probs - outcomes) ** 2)
        
        n_bins = min(10, len(valid_data) // 5)
        if n_bins < 2:
            n_bins = 2
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_by_bin = []
        calibration_errors = []
        
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
            
            if np.sum(mask) > 0:
                bin_probs = probs[mask]
                bin_outcomes = outcomes[mask]
                
                predicted_prob = np.mean(bin_probs)
                actual_rate = np.mean(bin_outcomes)
                error = abs(predicted_prob - actual_rate)
                
                calibration_by_bin.append({
                    "bin_range": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                    "n_samples": int(np.sum(mask)),
                    "predicted_prob": float(predicted_prob),
                    "actual_rate": float(actual_rate),
                    "error": float(error),
                })
                calibration_errors.append(error)
        
        avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 0
        
        return {
            "brier_score": float(brier_score),
            "avg_calibration_error": float(avg_calibration_error),
            "calibration_by_bin": calibration_by_bin,
            "n_samples": len(valid_data),
            "base_rate": float(np.mean(outcomes)),
        }
    
    def _calculate_ev_gap(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula la diferencia entre EV estimado y EV real.
        
        EV_estimated = EVS_adj promedio al entrar
        EV_real = PnL promedio realizado
        """
        valid_data = [
            d for d in data
            if d.get("evs_adj") is not None and d.get("actual_pnl") is not None
        ]
        
        if not valid_data:
            return {"error": "No valid data"}
        
        evs_values = np.array([d["evs_adj"] for d in valid_data])
        pnl_values = np.array([d["actual_pnl"] for d in valid_data])
        
        ev_estimated = np.mean(evs_values)
        ev_real = np.mean(pnl_values)
        ev_gap = ev_estimated - ev_real
        
        correlation = np.corrcoef(evs_values, pnl_values)[0, 1] if len(valid_data) > 2 else 0
        
        return {
            "ev_estimated": float(ev_estimated),
            "ev_real": float(ev_real),
            "ev_gap": float(ev_gap),
            "ev_gap_pct": float(ev_gap / ev_estimated * 100) if ev_estimated != 0 else 0,
            "evs_pnl_correlation": float(correlation) if not np.isnan(correlation) else 0,
            "n_samples": len(valid_data),
        }
    
    async def generate_calibration_report(self, days: int = 30) -> str:
        """
        v2.0: Genera reporte de calibración en texto.
        """
        calibration = await self.calculate_calibration_metrics(days)
        
        lines = [
            "=" * 50,
            f"MODEL CALIBRATION REPORT - Last {days} days",
            "=" * 50,
            "",
        ]
        
        if calibration.get("status") != "ok":
            lines.append(f"Status: {calibration.get('message', 'Insufficient data')}")
            return "\n".join(lines)
        
        lines.append(f"Total Samples: {calibration.get('n_samples', 0)}")
        lines.append("")
        
        p_rug = calibration.get("p_rug_calibration", {})
        lines.extend([
            "--- P_rug Calibration ---",
            f"Brier Score: {p_rug.get('brier_score', 0):.4f} (lower is better)",
            f"Avg Calibration Error: {p_rug.get('avg_calibration_error', 0):.1%}",
            f"Base Rate: {p_rug.get('base_rate', 0):.1%}",
            "",
        ])
        
        for bin_data in p_rug.get("calibration_by_bin", []):
            lines.append(
                f"  {bin_data['bin_range']}: "
                f"pred={bin_data['predicted_prob']:.1%}, "
                f"actual={bin_data['actual_rate']:.1%}, "
                f"n={bin_data['n_samples']}"
            )
        
        lines.append("")
        
        p_pump = calibration.get("p_pump_calibration", {})
        lines.extend([
            "--- P_pump Calibration ---",
            f"Brier Score: {p_pump.get('brier_score', 0):.4f}",
            f"Avg Calibration Error: {p_pump.get('avg_calibration_error', 0):.1%}",
            f"Base Rate: {p_pump.get('base_rate', 0):.1%}",
            "",
        ])
        
        for bin_data in p_pump.get("calibration_by_bin", []):
            lines.append(
                f"  {bin_data['bin_range']}: "
                f"pred={bin_data['predicted_prob']:.1%}, "
                f"actual={bin_data['actual_rate']:.1%}, "
                f"n={bin_data['n_samples']}"
            )
        
        lines.append("")
        
        ev_gap = calibration.get("ev_gap", {})
        lines.extend([
            "--- EV Gap Analysis ---",
            f"EV Estimated (avg EVS_adj): {ev_gap.get('ev_estimated', 0):.4f}",
            f"EV Real (avg PnL): {ev_gap.get('ev_real', 0):.4f}",
            f"EV Gap: {ev_gap.get('ev_gap', 0):.4f} ({ev_gap.get('ev_gap_pct', 0):.1f}%)",
            f"EVS-PnL Correlation: {ev_gap.get('evs_pnl_correlation', 0):.2f}",
            "",
        ])
        
        lines.extend([
            "--- Interpretation ---",
        ])
        
        brier_rug = p_rug.get('brier_score', 1)
        brier_pump = p_pump.get('brier_score', 1)
        
        if brier_rug < 0.25 and brier_pump < 0.25:
            lines.append("  Probability estimates are well-calibrated")
        elif brier_rug < 0.35 and brier_pump < 0.35:
            lines.append("  Probability estimates are reasonably calibrated")
        else:
            lines.append("  Probability estimates need improvement")
        
        ev_gap_val = abs(ev_gap.get('ev_gap', 0))
        if ev_gap_val < 0.02:
            lines.append("  EV estimates closely match reality")
        elif ev_gap_val < 0.05:
            lines.append("  EV estimates have moderate gap from reality")
        else:
            lines.append("  EV estimates significantly differ from reality - consider recalibration")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
