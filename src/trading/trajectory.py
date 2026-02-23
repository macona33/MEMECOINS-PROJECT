"""
Trajectory Monitor - Monitoreo de trayectorias de precio.
Rastrea MFE/MAE y gestiona actualizaciones en tiempo real.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from loguru import logger

from src.storage import DatabaseManager, TimeseriesManager
from src.data_sources import DexScreenerClient
from .simulator import TradeSimulator, SimulatedTrade
from config.settings import SETTINGS


class TrajectoryMonitor:
    """
    Monitor de trayectorias de precio para trades activos.
    
    Responsabilidades:
    - Polling de precios para trades activos
    - Cálculo continuo de MFE y MAE
    - Detección de condiciones de salida
    - Almacenamiento de series de precio
    """
    
    def __init__(
        self,
        db: DatabaseManager,
        simulator: TradeSimulator,
        timeseries: Optional[TimeseriesManager] = None,
        dex_client: Optional[DexScreenerClient] = None,
    ):
        self.db = db
        self.simulator = simulator
        self.timeseries = timeseries or TimeseriesManager()
        self.dex_client = dex_client or DexScreenerClient()
        
        self._running = False
        self._poll_interval = SETTINGS["price_poll_seconds"]
        self._callbacks: List[Callable] = []
    
    def on_price_update(self, callback: Callable) -> None:
        """Registra callback para actualizaciones de precio."""
        self._callbacks.append(callback)
    
    async def _notify(self, data: Dict[str, Any]) -> None:
        for callback in self._callbacks:
            try:
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in price callback: {e}")
    
    async def poll_prices(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene precios actuales para todos los trades activos.
        """
        trades = self.simulator.get_active_trades()
        
        if not trades:
            return {}
        
        addresses = [t.token_address for t in trades]
        prices = await self.dex_client.get_prices_batch(addresses)
        
        results = {}
        for trade in trades:
            addr = trade.token_address
            price = prices.get(addr, 0)
            
            if price > 0:
                results[addr] = {
                    "token_address": addr,
                    "symbol": trade.symbol,
                    "price": price,
                    "entry_price": trade.entry_price,
                    "pnl_pct": (price - trade.entry_price) / trade.entry_price,
                    "mfe": trade.current_mfe,
                    "mae": trade.current_mae,
                    "timestamp": datetime.now(),
                }
        
        return results
    
    async def update_single_trade(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Actualiza un trade específico con precio actual.
        """
        trade = self.simulator.get_trade(token_address)
        if not trade:
            return None
        
        prices = await self.dex_client.get_prices_batch([token_address])
        price = prices.get(token_address, 0)
        
        if price <= 0:
            logger.warning(f"Could not get price for {token_address}")
            return None
        
        close_reason = await self.simulator.update_trade(
            token_address=token_address,
            current_price=price,
        )
        
        price_record = {
            "token_address": token_address,
            "timestamp": datetime.now(),
            "price_usd": price,
            "volume": None,
            "liquidity": None,
            "market_cap": None,
        }
        await self.timeseries.append_prices([price_record])
        
        return {
            "token_address": token_address,
            "price": price,
            "pnl_pct": trade.pnl_pct,
            "mfe": trade.current_mfe,
            "mae": trade.current_mae,
            "close_reason": close_reason,
        }
    
    async def run_poll_cycle(self) -> Dict[str, Any]:
        """
        Ejecuta un ciclo completo de polling.
        """
        trades = self.simulator.get_active_trades()
        
        if not trades:
            return {"trades_updated": 0, "trades_closed": 0}
        
        updates = []
        closed = []
        
        for trade in trades:
            try:
                result = await self.update_single_trade(trade.token_address)
                
                if result:
                    updates.append(result)
                    
                    if result.get("close_reason"):
                        closed.append(result)
                    
                    await self._notify(result)
                
            except Exception as e:
                logger.error(f"Error updating trade {trade.token_address}: {e}")
        
        return {
            "trades_updated": len(updates),
            "trades_closed": len(closed),
            "closed_trades": closed,
        }
    
    async def run(self, poll_interval: int = None) -> None:
        """
        Ejecuta el monitor en loop continuo.
        """
        interval = poll_interval or self._poll_interval
        self._running = True
        
        logger.info(f"Starting trajectory monitor (interval: {interval}s)")
        
        while self._running:
            try:
                result = await self.run_poll_cycle()
                
                if result["trades_closed"]:
                    logger.info(f"Closed {result['trades_closed']} trades this cycle")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Trajectory monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trajectory monitor: {e}")
                await asyncio.sleep(interval)
    
    def stop(self) -> None:
        """Detiene el monitor."""
        self._running = False
        logger.info("Trajectory monitor stopped")
    
    async def get_trade_trajectory(
        self, 
        token_address: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Obtiene trayectoria completa de un trade.
        """
        import pandas as pd
        
        df = await self.timeseries.get_token_prices(
            token_address,
            start_date=datetime.now() - timedelta(hours=hours)
        )
        
        trade = self.simulator.get_trade(token_address)
        
        if df.empty:
            return {
                "token_address": token_address,
                "datapoints": 0,
                "mfe": trade.current_mfe if trade else 0,
                "mae": trade.current_mae if trade else 0,
            }
        
        entry_price = trade.entry_price if trade else df["price_usd"].iloc[0]
        
        df["pnl_pct"] = (df["price_usd"] - entry_price) / entry_price
        
        return {
            "token_address": token_address,
            "datapoints": len(df),
            "entry_price": entry_price,
            "current_price": df["price_usd"].iloc[-1],
            "mfe": df["pnl_pct"].max(),
            "mae": df["pnl_pct"].min(),
            "volatility": df["price_usd"].pct_change().std(),
            "price_range": {
                "min": df["price_usd"].min(),
                "max": df["price_usd"].max(),
            },
            "trajectory": df[["timestamp", "price_usd", "pnl_pct"]].to_dict("records"),
        }
    
    async def calculate_realized_volatility(
        self,
        token_address: str,
        window_minutes: int = 60
    ) -> Optional[float]:
        """
        Calcula volatilidad realizada basada en datos recientes.
        """
        df = await self.timeseries.get_token_prices(
            token_address,
            start_date=datetime.now() - timedelta(minutes=window_minutes)
        )
        
        if len(df) < 5:
            return None
        
        returns = df["price_usd"].pct_change().dropna()
        return returns.std()
    
    async def adjust_stops_dynamically(self, token_address: str) -> bool:
        """
        Ajusta stops basándose en volatilidad realizada.
        """
        trade = self.simulator.get_trade(token_address)
        if not trade or not trade.is_open:
            return False
        
        vol = await self.calculate_realized_volatility(token_address)
        if vol is None:
            return False
        
        base_stop = SETTINGS["base_stop_loss"]
        new_stop_pct = base_stop + vol * 2
        new_stop_pct = min(new_stop_pct, 0.30)
        
        new_stop_price = trade.entry_price * (1 - new_stop_pct)
        
        if trade.current_mfe > 0.10:
            lock_in = trade.current_mfe * 0.3
            trailing_stop = trade.entry_price * (1 + lock_in)
            new_stop_price = max(new_stop_price, trailing_stop)
        
        if new_stop_price != trade.stop_price:
            trade.stop_price = new_stop_price
            await self.db.update_active_trade(token_address, {
                "stop_price": new_stop_price
            })
            logger.debug(f"Adjusted stop for {trade.symbol} to ${new_stop_price:.8f}")
            return True
        
        return False
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Genera resumen del estado del monitoreo.
        """
        trades = self.simulator.get_active_trades()
        
        summary = {
            "active_trades": len(trades),
            "total_mfe": 0,
            "total_mae": 0,
            "avg_pnl_pct": 0,
            "trades": [],
        }
        
        if not trades:
            return summary
        
        total_pnl = 0
        
        for trade in trades:
            summary["total_mfe"] += trade.current_mfe
            summary["total_mae"] += trade.current_mae
            total_pnl += trade.pnl_pct
            
            summary["trades"].append({
                "symbol": trade.symbol,
                "pnl_pct": trade.pnl_pct,
                "mfe": trade.current_mfe,
                "mae": trade.current_mae,
                "duration_min": trade.duration_minutes,
            })
        
        n = len(trades)
        summary["avg_mfe"] = summary["total_mfe"] / n
        summary["avg_mae"] = summary["total_mae"] / n
        summary["avg_pnl_pct"] = total_pnl / n
        
        return summary
    
    async def close(self) -> None:
        """Cierra el monitor y limpia recursos."""
        self.stop()
        await self.dex_client.close()
