"""
Script para ver analíticas detalladas de cada trade.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import DatabaseManager


async def view_trades(days: int = 30):
    """Muestra todos los trades con detalles."""
    
    db = DatabaseManager()
    await db.connect()
    
    try:
        trades = await db.get_trade_history(days=days)
        
        if not trades:
            print("\nNo hay trades registrados en el periodo.\n")
            return
        
        print(f"\n{'='*80}")
        print(f"  HISTORIAL DE TRADES - Ultimos {days} dias")
        print(f"  Total: {len(trades)} trades")
        print(f"{'='*80}\n")
        
        for i, trade in enumerate(trades, 1):
            entry_time_raw = trade.get("entry_time", "N/A")
            exit_time_raw = trade.get("exit_time", "N/A")
            entry_time = entry_time_raw
            exit_time = exit_time_raw
            
            if isinstance(entry_time_raw, str) and entry_time_raw != "N/A":
                try:
                    entry_dt = datetime.fromisoformat(str(entry_time_raw).replace("Z", "+00:00"))
                    entry_time = entry_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            if isinstance(exit_time_raw, str) and exit_time_raw != "N/A":
                try:
                    exit_dt = datetime.fromisoformat(str(exit_time_raw).replace("Z", "+00:00"))
                    exit_time = exit_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            pnl_pct = trade.get("pnl_pct", 0) or 0
            pnl_usd = trade.get("pnl_usd", 0) or 0
            label = trade.get("label", "N/A")
            
            if pnl_pct > 0:
                pnl_indicator = "+"
                result_emoji = "[WIN]"
            elif pnl_pct < 0:
                pnl_indicator = ""
                result_emoji = "[LOSS]"
            else:
                pnl_indicator = ""
                result_emoji = "[EVEN]"
            
            real_duration = None
            if entry_time_raw and exit_time_raw:
                try:
                    entry_dt = datetime.fromisoformat(str(entry_time_raw).replace("Z", "+00:00"))
                    exit_dt = datetime.fromisoformat(str(exit_time_raw).replace("Z", "+00:00"))
                    real_duration = (exit_dt - entry_dt).total_seconds() / 60
                except:
                    pass
            
            duration_display = real_duration if (real_duration is not None) else trade.get("duration_minutes")
            
            print(f"--- Trade #{i} {result_emoji} ---")
            print(f"  Token:          {trade.get('token_address', 'N/A')[:20]}...")
            print(f"  Entry:          {entry_time}")
            print(f"  Exit:           {exit_time}")
            if duration_display is None:
                print("  Duration:       N/A")
            else:
                print(f"  Duration:       {float(duration_display):.1f} minutos")
            print(f"  ")
            print(f"  Entry Price:    ${trade.get('entry_price', 0):.10f}")
            print(f"  Exit Price:     ${trade.get('exit_price', 0):.10f}")
            print(f"  Position Size:  ${trade.get('position_size_usd', 0):.2f}")
            print(f"  Kelly Fraction: {trade.get('kelly_fraction', 0):.1%}")
            print(f"  ")
            print(f"  PnL:            {pnl_indicator}{pnl_pct:.1%} (${pnl_indicator}{pnl_usd:.2f})")
            print(f"  MFE:            +{trade.get('mfe', 0):.1%} (maximo alcanzado)")
            print(f"  MAE:            {trade.get('mae', 0):.1%} (minimo alcanzado)")
            print(f"  ")
            print(f"  Exit Reason:    {trade.get('exit_reason', 'N/A')}")
            print(f"  Label:          {label.upper()}")
            print(f"  ")
            print(f"  --- Scores al Entrar ---")
            print(f"  EVS_adj:        {trade.get('evs_at_entry', 0):.4f}")
            print(f"  P_rug:          {trade.get('p_rug_at_entry', 0):.1%}")
            print(f"  P_pump:         {trade.get('p_pump_at_entry', 0):.1%}")
            print(f"\n")
        
        print(f"{'='*80}")
        print("  RESUMEN")
        print(f"{'='*80}")
        
        total_pnl = sum(t.get("pnl_usd", 0) or 0 for t in trades)
        wins = [t for t in trades if (t.get("pnl_pct", 0) or 0) > 0]
        losses = [t for t in trades if (t.get("pnl_pct", 0) or 0) < 0]
        pumps = [t for t in trades if t.get("label") == "pump"]
        rugs = [t for t in trades if t.get("label") == "rug"]
        
        print(f"  Trades Ganadores:  {len(wins)}")
        print(f"  Trades Perdedores: {len(losses)}")
        print(f"  Win Rate:          {len(wins)/len(trades):.1%}")
        print(f"  ")
        print(f"  Labels PUMP:       {len(pumps)}")
        print(f"  Labels RUG:        {len(rugs)}")
        print(f"  Hit Rate:          {len(pumps)/len(trades):.1%}")
        print(f"  ")
        print(f"  PnL Total:         ${total_pnl:.2f}")
        
        if wins:
            avg_win = sum(t.get("pnl_pct", 0) or 0 for t in wins) / len(wins)
            print(f"  Avg Win:           +{avg_win:.1%}")
        if losses:
            avg_loss = sum(t.get("pnl_pct", 0) or 0 for t in losses) / len(losses)
            print(f"  Avg Loss:          {avg_loss:.1%}")
        
        print(f"\n{'='*80}\n")
        
    finally:
        await db.close()


async def view_single_trade(trade_id: int):
    """Muestra un trade específico por ID."""
    
    db = DatabaseManager()
    await db.connect()
    
    try:
        cursor = await db._connection.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        )
        trade = await cursor.fetchone()
        
        if not trade:
            print(f"\nTrade #{trade_id} no encontrado.\n")
            return
        
        trade = dict(trade)
        
        print(f"\n{'='*60}")
        print(f"  DETALLE TRADE #{trade_id}")
        print(f"{'='*60}\n")
        
        for key, value in trade.items():
            if value is not None:
                print(f"  {key}: {value}")
        
        print(f"\n{'='*60}\n")
        
    finally:
        await db.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ver historial de trades")
    parser.add_argument("--days", type=int, default=30, help="Dias de historial")
    parser.add_argument("--id", type=int, help="Ver trade especifico por ID")
    
    args = parser.parse_args()
    
    if args.id:
        asyncio.run(view_single_trade(args.id))
    else:
        asyncio.run(view_trades(days=args.days))


if __name__ == "__main__":
    main()
