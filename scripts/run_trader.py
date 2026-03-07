"""
Script principal de paper trading v2.0.
Sistema paramétrico adaptativo con edge estable.
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.storage import DatabaseManager, TimeseriesManager
from src.data_sources import DexScreenerClient, SolscanClient
from src.ingestion import TokenScanner, TokenFilter, FeatureExtractor
from src.models import HazardModel, PumpModel, EVSCalculator
from src.trading import KellyCalculator, TradeSimulator, TrajectoryMonitor, RiskManager
from src.notifications import notify_trade_opened, notify_trade_closed
from src.calibration import LabelGenerator, ModelUpdater, MetricsTracker
from src.market import RegimeDetector
from config.settings import SETTINGS

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    "logs/trader_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


class TradingApp:
    """Aplicación principal de paper trading v2.0."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.timeseries = TimeseriesManager()
        self.dex_client = DexScreenerClient()
        self.solscan_client = SolscanClient()
        
        self.scanner = TokenScanner(
            db=self.db,
            dex_client=self.dex_client,
            solscan_client=self.solscan_client,
        )
        
        self.filter = TokenFilter()
        self.feature_extractor = FeatureExtractor(self.db, self.timeseries)
        
        self.hazard_model = HazardModel()
        self.pump_model = PumpModel()
        self.evs_calculator = EVSCalculator(self.hazard_model, self.pump_model)
        
        self.risk_manager = RiskManager()
        self.regime_detector = RegimeDetector()
        
        self.kelly_calculator = KellyCalculator()
        self.simulator = TradeSimulator(
            db=self.db,
            dex_client=self.dex_client,
            evs_calculator=self.evs_calculator,
            kelly_calculator=self.kelly_calculator,
        )
        self.trajectory_monitor = TrajectoryMonitor(
            db=self.db,
            simulator=self.simulator,
            timeseries=self.timeseries,
            dex_client=self.dex_client,
        )
        
        self.model_updater = ModelUpdater(
            db=self.db,
            hazard_model=self.hazard_model,
            pump_model=self.pump_model,
            evs_calculator=self.evs_calculator,
        )
        self.metrics_tracker = MetricsTracker(self.db)
        
        self._running = False
        self._last_recalibration = datetime.now()
        self._last_regime_check = datetime.now()
        self._recalibration_interval = timedelta(
            hours=SETTINGS.get("recalibration_interval_hours", 24)
        )
        self._trades_since_recalibration = 0
    
    async def initialize(self) -> None:
        """Inicializa todos los componentes v2.0."""
        logger.info("Initializing trading application v2.0...")
        
        Path("logs").mkdir(exist_ok=True)
        
        await self.db.connect()
        
        self.hazard_model.set_database(self.db)
        self.pump_model.set_database(self.db)
        await self.hazard_model.load_params()
        await self.pump_model.load_params()
        
        self.risk_manager.set_database(self.db)
        await self.risk_manager.load_state()
        
        self.regime_detector.set_database(self.db)
        await self.regime_detector.load_state()
        
        trade_count = await self.db.get_trade_count()
        self.kelly_calculator.set_trade_count(trade_count)
        logger.info(f"v2.0: Trade count for Kelly adjustment: {trade_count}")
        
        self.kelly_calculator.set_gamma_multiplier(
            self.risk_manager.get_gamma_multiplier()
        )
        
        await self.scanner.initialize()
        await self.simulator.load_active_trades()
        self.simulator.set_capital(self.risk_manager.state.current_equity)

        self.scanner.on_new_token(self.process_token)
        self.simulator.on_trade_event(self.on_trade_event)
        self.trajectory_monitor.on_price_update(self.on_price_update)
        
        logger.info("Trading application v2.0 initialized")
        equity = self.risk_manager.state.current_equity
        logger.info(
            f"Capital: ${equity:,.2f}"
            + (f" (inicial: ${SETTINGS['initial_capital']:,})" if equity != SETTINGS["initial_capital"] else "")
        )
        logger.info(f"Kelly gamma: {SETTINGS['kelly_gamma']} (v2.0)")
        logger.info(f"Max position: {SETTINGS['max_position_pct']:.1%} (v2.0)")
        logger.info(f"Active trades: {self.simulator.active_trade_count}")
        logger.info(f"Market regime: {self.regime_detector.current_regime.value}")
        logger.info(f"Risk level: {self.risk_manager.get_risk_level()}")
    
    async def process_token(self, token: dict) -> None:
        """Procesa token nuevo y abre trade si cumple criterios v2.0."""
        symbol = token.get("symbol", "UNKNOWN")
        address = token.get("address", "")
        
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.debug(f"v2.0 Risk block: {reason}")
            return
        
        filter_result = self.filter.apply_filters(token)
        if not filter_result.passed:
            return
        
        features = await self.feature_extractor.extract_and_save(token)
        evs_result = self.evs_calculator.calculate(features)
        await self.evs_calculator.save_scores(self.db, address, features)
        
        adjustments = self.regime_detector.get_adjustments()
        adjusted_threshold = adjustments.evs_threshold
        
        if evs_result.evs_adj < adjusted_threshold:
            logger.debug(
                f"v2.0 Regime filter: {symbol} EVS={evs_result.evs_adj:.4f} "
                f"< threshold={adjusted_threshold:.4f} ({adjustments.regime.value})"
            )
            return
        
        if not evs_result.is_tradeable:
            return
        
        logger.info(
            f"Tradeable token found: {symbol} "
            f"EVS_adj={evs_result.evs_adj:.4f} "
            f"[{self.regime_detector.current_regime.value}]"
        )
        
        self.kelly_calculator.set_gamma_multiplier(
            self.risk_manager.get_gamma_multiplier()
        )
        
        trade = await self.simulator.open_trade(
            token_address=address,
            token_info=token,
            evs_result=evs_result,
            features=features,
        )
        
        if trade:
            await self._save_trade_features(trade.trade_id, address, features, evs_result)
            notify_trade_opened(trade)

            logger.info(
                f"Opened trade: {symbol} @ ${trade.entry_price:.8f}, "
                f"Size: ${trade.position_size_usd:.2f}, "
                f"Kelly mult: {self.risk_manager.get_gamma_multiplier():.2f}"
            )
    
    async def _save_trade_features(self, trade_id: int, token_address: str, features: dict, evs_result) -> None:
        """v2.0: Guarda features completas para recalibración."""
        if trade_id is None:
            return
        
        stops = self.kelly_calculator.calculate_stop_levels(
            entry_price=1.0,
            evs_result=evs_result,
            features=features,
        )
        
        scores = {
            "p_rug": evs_result.p_rug,
            "p_pump": evs_result.p_pump,
            "expected_g": evs_result.expected_g,
            "evs": evs_result.evs,
            "evs_adj": evs_result.evs_adj,
            "sigma": evs_result.sigma_token,
        }
        
        await self.db.save_trade_features(
            trade_id=trade_id,
            token_address=token_address,
            features=features,
            scores=scores,
            stops=stops,
        )
    
    async def on_trade_event(self, event: str, trade) -> None:
        """Callback para eventos de trades v2.0."""
        if event == "trade_closed":
            notify_trade_closed(trade)
            is_win = trade.pnl_usd > 0
            self.risk_manager.record_trade_result(trade.pnl_usd, is_win)
            
            self._trades_since_recalibration += 1
            
            await self._update_trade_features_outcome(trade)
            
            logger.info(
                f"Trade closed: {trade.symbol}, "
                f"PnL: {trade.pnl_pct:.1%} (${trade.pnl_usd:.2f}), "
                f"Reason: {trade.exit_reason}"
            )
            
            summary = self.simulator.get_portfolio_summary()
            risk_status = self.risk_manager.get_status()
            
            logger.info(
                f"Portfolio: {summary['active_trades']} trades, "
                f"Total PnL: ${summary['total_pnl_usd']:.2f}, "
                f"Drawdown: {risk_status['current_drawdown']}, "
                f"Risk: {self.risk_manager.get_risk_level()}"
            )
            
            await self.risk_manager.save_state()
            
            if self._should_trigger_recalibration():
                asyncio.create_task(self._run_recalibration())
    
    async def _update_trade_features_outcome(self, trade) -> None:
        """v2.0: Actualiza outcome en trade_features."""
        trade_id = getattr(trade, "trade_id", None) or await self.db.get_last_trade_id(trade.token_address)
        if trade_id is None:
            return
        
        stop_executed = trade.exit_reason in ["stop_loss", "trailing_stop"]
        # v2.0: Usar label path-dependent (pump/rug/neutral) para recalibración, no exit_reason
        label = getattr(trade, "label", None) or trade.exit_reason or "unknown"

        await self.db.update_trade_features_outcome(
            trade_id=trade_id,
            mfe=trade.current_mfe,
            mae=trade.current_mae,
            pnl=trade.pnl_pct,
            stop_executed=stop_executed,
            label=label,
        )
    
    def _should_trigger_recalibration(self) -> bool:
        """v2.0: Determina si se debe ejecutar recalibración."""
        new_trades_trigger = SETTINGS.get("new_trades_trigger", 30)
        if self._trades_since_recalibration >= new_trades_trigger:
            return True
        
        if datetime.now() - self._last_recalibration > self._recalibration_interval:
            return True
        
        return False
    
    async def _run_recalibration(self) -> None:
        """v2.0: Ejecuta recalibración de modelos."""
        logger.info("v2.0: Running model recalibration...")
        
        try:
            trade_features = await self.db.get_training_dataset(
                days=SETTINGS.get("recalibration_window_days", 7)
            )
            
            if len(trade_features) < SETTINGS.get("min_trades_for_recalibration", 30):
                logger.info(
                    f"v2.0: Insufficient data for recalibration "
                    f"({len(trade_features)} trades)"
                )
                return
            
            hazard_trainer = self.hazard_model.get_trainer()
            hazard_result = hazard_trainer.train(trade_features)
            if hazard_result.success:
                self.hazard_model.update_from_trainer()
                await self.hazard_model.save_params()
                logger.info(
                    f"v2.0: Hazard model updated, "
                    f"loss improved {hazard_result.improvement:.1%}"
                )
            
            pump_trainer = self.pump_model.get_trainer()
            pump_result = pump_trainer.train(trade_features)
            if pump_result.success:
                self.pump_model.update_from_trainer()
                await self.pump_model.save_params()
                logger.info(
                    f"v2.0: Pump model updated, "
                    f"loss improved {pump_result.improvement:.1%}"
                )
            
            g_update = await self.pump_model.update_g_buckets(trade_features)
            logger.info(f"v2.0: G buckets updated: {g_update.get('updated', [])}")
            
            self._last_recalibration = datetime.now()
            self._trades_since_recalibration = 0
            
        except Exception as e:
            logger.error(f"v2.0: Recalibration error: {e}")
    
    async def on_price_update(self, data: dict) -> None:
        """Callback para actualizaciones de precio."""
        if data.get("close_reason"):
            logger.info(
                f"Price trigger: {data.get('symbol', data['token_address'][:8])}, "
                f"Reason: {data['close_reason']}"
            )
    
    async def run_scanner_task(self) -> None:
        """Tarea de escaneo de tokens."""
        try:
            await self.scanner.run()
        except asyncio.CancelledError:
            pass
    
    async def run_monitor_task(self) -> None:
        """Tarea de monitoreo de trayectorias."""
        try:
            await self.trajectory_monitor.run()
        except asyncio.CancelledError:
            pass
    
    async def run_maintenance_task(self) -> None:
        """Tarea de mantenimiento periódico v2.0."""
        while self._running:
            try:
                await asyncio.sleep(1800)
                
                if self.regime_detector.should_check():
                    recent_tokens = await self.db.get_recent_tokens(hours=2)
                    adjustments = await self.regime_detector.update(recent_tokens)
                    logger.info(
                        f"v2.0: Regime updated: {adjustments.regime.value}, "
                        f"EVS threshold={adjustments.evs_threshold:.4f}"
                    )
                
                if self._should_trigger_recalibration():
                    await self._run_recalibration()
                
                await self.timeseries.cleanup_old_data(days_to_keep=30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance error: {e}")
    
    async def run_status_task(self) -> None:
        """Tarea de reporte de estado periódico v2.0."""
        while self._running:
            try:
                await asyncio.sleep(300)
                
                summary = self.simulator.get_portfolio_summary()
                risk_status = self.risk_manager.get_status()
                regime = self.regime_detector.current_regime.value
                
                logger.info(
                    f"Status: {summary['active_trades']} trades, "
                    f"Allocated: ${summary['total_allocated']:.2f}, "
                    f"PnL: ${summary['total_pnl_usd']:.2f} ({summary['total_pnl_pct']:.1%}), "
                    f"Regime: {regime}, "
                    f"Risk: {self.risk_manager.get_risk_level()}"
                )
                
                if risk_status['is_frozen']:
                    logger.warning(
                        f"v2.0: Trading FROZEN until {risk_status['frozen_until']}"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Status task error: {e}")

    async def run(self) -> None:
        """Ejecuta todas las tareas en paralelo."""
        self._running = True
        
        logger.info("Starting paper trading system...")
        
        tasks = [
            asyncio.create_task(self.run_scanner_task(), name="scanner"),
            asyncio.create_task(self.run_monitor_task(), name="monitor"),
            asyncio.create_task(self.run_maintenance_task(), name="maintenance"),
            asyncio.create_task(self.run_status_task(), name="status"),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Cierra todos los recursos."""
        logger.info("Shutting down trading system...")
        
        self._running = False
        self.scanner.stop()
        self.trajectory_monitor.stop()
        
        await self.scanner.close()
        await self.trajectory_monitor.close()
        await self.simulator.close()
        await self.db.close()
        
        logger.info("Trading system shutdown complete")
    
    def handle_signal(self, sig) -> None:
        """Maneja señales de terminación."""
        logger.info(f"Received signal {sig}, shutting down...")
        self._running = False


async def main():
    """Punto de entrada principal."""
    app = TradingApp()
    
    loop = asyncio.get_event_loop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: app.handle_signal(s))
        except NotImplementedError:
            pass
    
    try:
        await app.initialize()
        await app.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    print("""
    ====================================================
    |   SOLANA MEMECOIN PAPER TRADING SYSTEM v2.0      |
    |                                                  |
    |   Adaptive system with stable edge               |
    |   Running in PAPER TRADING mode - No real money  |
    |   Press Ctrl+C to stop                           |
    ====================================================
    
    v2.0 Features:
    - Incremental logistic regression for P_rug/P_pump
    - Dynamic G based on historical MFE
    - Market regime detection (LOW/NORMAL/HIGH)
    - Dynamic drawdown control
    - Conservative Kelly (gamma=0.10)
    - Calibration metrics (Brier score, EV gap)
    """)
    
    asyncio.run(main())
