"""
Script principal para ejecutar el scanner de tokens.
Detecta tokens nuevos y los procesa a través del pipeline completo.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.storage import DatabaseManager, TimeseriesManager
from src.data_sources import DexScreenerClient, SolscanClient
from src.ingestion import TokenScanner, TokenFilter, FeatureExtractor
from src.models import HazardModel, PumpModel, EVSCalculator
from config.settings import SETTINGS

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    "logs/scanner_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


class ScannerApp:
    """Aplicación principal del scanner."""
    
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
        
        self._running = False
    
    async def initialize(self) -> None:
        """Inicializa todos los componentes."""
        logger.info("Initializing scanner application...")
        
        Path("logs").mkdir(exist_ok=True)
        
        await self.db.connect()
        
        self.hazard_model.set_database(self.db)
        self.pump_model.set_database(self.db)
        await self.hazard_model.load_params()
        await self.pump_model.load_params()
        
        await self.scanner.initialize()
        
        self.scanner.on_new_token(self.process_token)
        
        logger.info("Scanner application initialized")
    
    async def process_token(self, token: dict) -> None:
        """
        Procesa un token nuevo detectado.
        Pipeline: Filtro -> Features -> Scoring -> Storage
        """
        symbol = token.get("symbol", "UNKNOWN")
        address = token.get("address", "")
        
        logger.info(f"Processing new token: {symbol}")
        
        filter_result = self.filter.apply_filters(token)
        
        if not filter_result.passed:
            logger.debug(f"Token {symbol} filtered out: {filter_result.rejection_reason}")
            return
        
        if filter_result.warnings:
            logger.warning(f"Token {symbol} warnings: {filter_result.warnings}")
        
        features = await self.feature_extractor.extract_and_save(token)
        
        evs_result = self.evs_calculator.calculate(features)
        
        await self.evs_calculator.save_scores(self.db, address, features)
        
        logger.info(
            f"Token {symbol} scored: "
            f"P_rug={evs_result.p_rug:.1%}, "
            f"P_pump={evs_result.p_pump:.1%}, "
            f"EVS_adj={evs_result.evs_adj:.4f}, "
            f"Tradeable={evs_result.is_tradeable}"
        )
        
        if evs_result.is_tradeable:
            logger.info(f"*** TRADEABLE OPPORTUNITY: {symbol} ***")
    
    async def run(self) -> None:
        """Ejecuta el scanner en loop continuo."""
        self._running = True
        
        logger.info("Starting scanner loop...")
        logger.info(f"Scan interval: {SETTINGS['scan_interval_seconds']}s")
        logger.info(f"Min liquidity: ${SETTINGS['min_liquidity_usd']:,}")
        
        try:
            await self.scanner.run()
        except asyncio.CancelledError:
            logger.info("Scanner cancelled")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Cierra todos los recursos."""
        logger.info("Shutting down scanner...")
        
        self.scanner.stop()
        await self.scanner.close()
        await self.db.close()
        
        logger.info("Scanner shutdown complete")
    
    def handle_signal(self, sig) -> None:
        """Maneja señales de terminación."""
        logger.info(f"Received signal {sig}, shutting down...")
        self._running = False
        self.scanner.stop()


async def main():
    """Punto de entrada principal."""
    app = ScannerApp()
    
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
    |     SOLANA MEMECOIN SCANNER - Paper Trading      |
    |                                                  |
    |  Scanning for new tokens on Solana DEXes...      |
    |  Press Ctrl+C to stop                            |
    ====================================================
    """)
    
    asyncio.run(main())
