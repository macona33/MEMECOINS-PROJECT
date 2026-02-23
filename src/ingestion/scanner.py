"""
Scanner de tokens nuevos en Solana.
Detecta tokens recién creados y los envía al pipeline de evaluación.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set, Optional, Callable
from loguru import logger

from src.data_sources import DexScreenerClient, SolscanClient
from src.storage import DatabaseManager
from config.settings import SETTINGS


class TokenScanner:
    """Scanner que detecta y procesa tokens nuevos."""
    
    def __init__(
        self,
        db: DatabaseManager,
        dex_client: Optional[DexScreenerClient] = None,
        solscan_client: Optional[SolscanClient] = None,
    ):
        self.db = db
        self.dex_client = dex_client or DexScreenerClient()
        self.solscan_client = solscan_client or SolscanClient()
        self._seen_tokens: Set[str] = set()
        self._running = False
        self._callbacks: List[Callable] = []
    
    def on_new_token(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Registra callback para nuevos tokens detectados."""
        self._callbacks.append(callback)
    
    async def _notify_callbacks(self, token: Dict[str, Any]) -> None:
        """Notifica a todos los callbacks registrados."""
        for callback in self._callbacks:
            try:
                result = callback(token)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in token callback: {e}")
    
    async def initialize(self) -> None:
        """Inicializa el scanner cargando tokens ya conocidos."""
        recent_tokens = await self.db.get_recent_tokens(hours=48)
        self._seen_tokens = {t["address"] for t in recent_tokens}
        logger.info(f"Scanner initialized with {len(self._seen_tokens)} known tokens")
    
    async def scan_once(self) -> List[Dict[str, Any]]:
        """
        Ejecuta un escaneo único buscando tokens nuevos.
        Retorna lista de tokens nuevos encontrados.
        Usa múltiples fuentes para maximizar descubrimiento.
        """
        new_tokens = []
        
        try:
            pairs = await self.dex_client.get_new_solana_pairs(
                min_liquidity=SETTINGS["min_liquidity_usd"]
            )
            
            if len(pairs) < 5:
                trending_pairs = await self.dex_client.get_trending_solana_pairs(
                    min_liquidity=SETTINGS["min_liquidity_usd"]
                )
                seen_in_pairs = {p.get("baseToken", {}).get("address") for p in pairs}
                for p in trending_pairs:
                    if p.get("baseToken", {}).get("address") not in seen_in_pairs:
                        pairs.append(p)
            
            logger.debug(f"Found {len(pairs)} pairs from DexScreener")
            
            for pair_data in pairs:
                parsed = self.dex_client.parse_token_data(pair_data)
                address = parsed.get("address")
                
                if not address:
                    continue
                
                if address in self._seen_tokens:
                    continue
                
                age_hours = parsed.get("age_hours")
                if age_hours is not None:
                    if age_hours < SETTINGS["min_token_age_minutes"] / 60:
                        continue
                    if age_hours > SETTINGS["max_token_age_hours"]:
                        continue
                
                self._seen_tokens.add(address)
                new_tokens.append(parsed)
                
                await self.db.insert_token(parsed)
                
                logger.info(
                    f"New token: {parsed.get('symbol')} ({address[:8]}...) "
                    f"Liq: ${parsed.get('liquidity_usd', 0):,.0f}"
                )
                
                await self._notify_callbacks(parsed)
        
        except Exception as e:
            logger.error(f"Error during scan: {e}")
        
        return new_tokens
    
    async def run(self, interval_seconds: int = None) -> None:
        """
        Ejecuta el scanner en loop continuo.
        """
        interval = interval_seconds or SETTINGS["scan_interval_seconds"]
        self._running = True
        
        logger.info(f"Starting token scanner (interval: {interval}s)")
        
        await self.initialize()
        
        while self._running:
            try:
                new_tokens = await self.scan_once()
                
                if new_tokens:
                    logger.info(f"Scan complete: {len(new_tokens)} new tokens")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Scanner cancelled")
                break
            except Exception as e:
                logger.error(f"Scanner error: {e}")
                await asyncio.sleep(interval)
    
    def stop(self) -> None:
        """Detiene el scanner."""
        self._running = False
        logger.info("Scanner stopped")
    
    async def close(self) -> None:
        """Cierra conexiones."""
        self.stop()
        await self.dex_client.close()
        await self.solscan_client.close()
    
    async def get_token_details(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene detalles completos de un token específico.
        """
        pairs = await self.dex_client.get_token_pairs(address)
        
        if not pairs:
            return None
        
        best_pair = max(
            pairs, 
            key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0
        )
        
        parsed = self.dex_client.parse_token_data(best_pair)
        enriched = await self.solscan_client.enrich_token_data(parsed)
        
        return enriched
    
    async def refresh_token(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Actualiza datos de un token existente.
        """
        token_data = await self.get_token_details(address)
        
        if token_data:
            await self.db.insert_token(token_data)
        
        return token_data
