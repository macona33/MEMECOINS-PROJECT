"""
Script para generar reportes diarios de rendimiento.
Puede ejecutarse manualmente o programarse con cron/scheduler.
"""

import asyncio
import sys
import json
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.storage import DatabaseManager
from src.models import HazardModel, PumpModel
from src.calibration import MetricsTracker, ModelUpdater, LabelGenerator
from src.notifications import send_daily_pnl_chart
from config.settings import SETTINGS

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)


async def generate_daily_report(days: int = 7):
    """Genera reporte completo de rendimiento."""
    
    db = DatabaseManager()
    await db.connect()
    
    try:
        metrics_tracker = MetricsTracker(db)
        
        hazard_model = HazardModel()
        pump_model = PumpModel()
        hazard_model.set_database(db)
        pump_model.set_database(db)
        await hazard_model.load_params()
        await pump_model.load_params()
        
        model_updater = ModelUpdater(
            db=db,
            hazard_model=hazard_model,
            pump_model=pump_model,
        )
        
        print("\n" + "=" * 60)
        print("       PAPER TRADING DAILY REPORT")
        print(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)
        
        today_metrics = await metrics_tracker.calculate_daily_metrics()
        
        print("\n[TODAY'S PERFORMANCE]")
        print("-" * 40)
        if today_metrics.get("total_trades", 0) > 0:
            print(f"  Trades Closed: {today_metrics['total_trades']}")
            print(f"  Hit Rate: {today_metrics.get('hit_rate', 0):.1%}")
            print(f"  Total PnL: {today_metrics.get('total_pnl_pct', 0):.1%} (${today_metrics.get('total_pnl_usd', 0):.2f})")
            print(f"  Pumps: {today_metrics.get('pump_count', 0)} | Rugs: {today_metrics.get('rug_count', 0)}")
        else:
            print("  No trades closed today")
        
        period_metrics = await metrics_tracker.calculate_period_metrics(days=days)
        
        print(f"\n[{days}-DAY PERIOD PERFORMANCE]")
        print("-" * 40)
        print(f"  Total Trades: {period_metrics.get('total_trades', 0)}")
        print(f"  Trades/Day: {period_metrics.get('trades_per_day', 0):.1f}")
        print(f"  Hit Rate: {period_metrics.get('hit_rate', 0):.1%}")
        print(f"  Win Rate: {period_metrics.get('win_rate', 0):.1%}")
        print(f"  Total PnL: {period_metrics.get('total_pnl_pct', 0):.1%} (${period_metrics.get('total_pnl_usd', 0):.2f})")
        print(f"  Sharpe Ratio: {period_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {period_metrics.get('max_drawdown', 0):.1%}")
        print(f"  Profit Factor: {period_metrics.get('profit_factor', 0):.2f}")
        
        print("\n[TRADE ANALYSIS]")
        print("-" * 40)
        print(f"  Avg Win: {period_metrics.get('avg_win', 0):.1%}")
        print(f"  Avg Loss: {period_metrics.get('avg_loss', 0):.1%}")
        print(f"  Best Trade: {period_metrics.get('best_trade', 0):.1%}")
        print(f"  Worst Trade: {period_metrics.get('worst_trade', 0):.1%}")
        print(f"  Pumps: {period_metrics.get('pump_count', 0)}")
        print(f"  Rugs: {period_metrics.get('rug_count', 0)}")
        
        success = await metrics_tracker.check_success_criteria()
        
        print("\n[SUCCESS CRITERIA]")
        print("-" * 40)
        for name, criterion in success.get("criteria", {}).items():
            status = "[PASS]" if criterion["passed"] else "[FAIL]"
            current = criterion["current"]
            target = criterion["target"]
            
            if isinstance(current, float):
                if current < 1:
                    current_str = f"{current:.1%}"
                else:
                    current_str = f"{current:.2f}"
            else:
                current_str = str(current)
            
            if isinstance(target, float):
                if target < 1:
                    target_str = f"{target:.1%}"
                else:
                    target_str = f"{target:.2f}"
            else:
                target_str = str(target)
            
            print(f"  {status} {name}: {current_str} (target: {target_str})")
        
        print(f"\n  Recommendation: {success.get('recommendation', 'N/A')}")
        
        if period_metrics.get("total_trades", 0) >= SETTINGS["min_trades_for_recalibration"]:
            print("\n[MODEL PERFORMANCE]")
            print("-" * 40)
            
            performance = await model_updater.evaluate_model_performance()
            
            hazard_acc = performance.get("hazard_model", {})
            print(f"  Hazard Model Accuracy: {hazard_acc.get('accuracy', 0):.1%}")
            print(f"  Hazard Precision: {hazard_acc.get('precision', 0):.1%}")
            print(f"  Hazard Recall: {hazard_acc.get('recall', 0):.1%}")
            
            pump_acc = performance.get("pump_model", {})
            print(f"  Pump Model Accuracy: {pump_acc.get('accuracy', 0):.1%}")
            print(f"  Pump Mean G Error: {pump_acc.get('mean_g_error', 0):.1%}")
            
            print("\n[v2.0 CALIBRATION METRICS]")
            print("-" * 40)
            
            calibration_report = await metrics_tracker.generate_calibration_report(days=days)
            for line in calibration_report.split("\n"):
                if line.strip() and not line.startswith("="):
                    print(f"  {line}")
        
        print("\n[CURRENT STATE]")
        print("-" * 40)
        
        active_trades = await db.get_active_trades()
        print(f"  Active Trades: {len(active_trades)}")
        
        if active_trades:
            total_allocated = sum(t.get("position_size_usd", 0) for t in active_trades)
            print(f"  Total Allocated: ${total_allocated:.2f}")
        
        recent_tokens = await db.get_recent_tokens(hours=24)
        print(f"  Tokens Scanned (24h): {len(recent_tokens)}")
        
        top_tokens = await db.get_top_ranked_tokens(limit=5)
        if top_tokens:
            print("\n[TOP RANKED TOKENS (Current)]")
            print("-" * 40)
            for i, token in enumerate(top_tokens, 1):
                print(
                    f"  {i}. {token.get('symbol', 'N/A')}: "
                    f"EVS={token.get('evs_adj', 0):.4f}, "
                    f"P_rug={token.get('p_rug', 0):.1%}"
                )
        
        print("\n" + "=" * 60)
        print("       END OF REPORT")
        print("=" * 60 + "\n")
        
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        report_data = await metrics_tracker.export_metrics(days=days)
        report_data["today"] = today_metrics
        report_data["success_criteria"] = success

        equity_curve = await metrics_tracker.get_equity_curve_from_trades(days=days)
        if not equity_curve:
            equity_curve = await metrics_tracker.get_equity_curve(days=days)
        await send_daily_pnl_chart(equity_curve, initial_capital=SETTINGS["initial_capital"])

        report_file = report_dir / f"report_{date.today().isoformat()}.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Full report saved to: {report_file}")
        
        return report_data
        
    finally:
        await db.close()


async def run_recalibration():
    """Ejecuta recalibración de modelos."""
    
    db = DatabaseManager()
    await db.connect()
    
    try:
        hazard_model = HazardModel()
        pump_model = PumpModel()
        hazard_model.set_database(db)
        pump_model.set_database(db)
        await hazard_model.load_params()
        await pump_model.load_params()
        
        model_updater = ModelUpdater(
            db=db,
            hazard_model=hazard_model,
            pump_model=pump_model,
        )
        
        print("\n🔄 Running model recalibration...")
        
        results = await model_updater.recalibrate_all()
        
        print("\n📊 RECALIBRATION RESULTS")
        print("-" * 40)
        
        if results["hazard"]["updated"]:
            print(f"  Hazard Model: Updated with {results['hazard']['samples']} samples")
            print(f"    Accuracy: {results['hazard']['accuracy'].get('accuracy', 0):.1%}")
        else:
            print(f"  Hazard Model: {results['hazard'].get('reason', 'Not updated')}")
        
        if results["pump"]["updated"]:
            print(f"  Pump Model: Updated with {results['pump']['samples']} samples")
            print(f"    Accuracy: {results['pump']['accuracy'].get('accuracy', 0):.1%}")
        else:
            print(f"  Pump Model: {results['pump'].get('reason', 'Not updated')}")
        
        if "threshold_suggestions" in results:
            print("\n📈 THRESHOLD SUGGESTIONS")
            for param, suggestion in results["threshold_suggestions"].items():
                print(
                    f"  {param}: {suggestion['current']:.4f} → {suggestion['suggested']:.4f}"
                    f" ({suggestion['reason']})"
                )
        
        print("\n✅ Recalibration complete")
        
    finally:
        await db.close()


def main():
    """Punto de entrada principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate trading reports")
    parser.add_argument(
        "--days", 
        type=int, 
        default=7, 
        help="Number of days for period metrics"
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Run model recalibration"
    )
    
    args = parser.parse_args()
    
    if args.recalibrate:
        asyncio.run(run_recalibration())
    else:
        asyncio.run(generate_daily_report(days=args.days))


if __name__ == "__main__":
    main()
