"""
Cliente async para DexScreener API.
Documentación: https://docs.dexscreener.com/api/reference
Rate limit: 300 requests/minute (gratuito)
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger

from config.settings import SETTINGS


@dataclass
class RateLimiter:
    """Rate limiter con ventana deslizante."""
    max_requests: int
    window_seconds: int = 60
    
    def __post_init__(self):
        self._timestamps: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            cutoff = now - self.window_seconds
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            
            if len(self._timestamps) >= self.max_requests:
                sleep_time = self._timestamps[0] - cutoff + 0.1
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                self._timestamps = self._timestamps[1:]
            
            self._timestamps.append(now)


class DexScreenerClient:
    """Cliente para DexScreener API con rate limiting."""
    
    BASE_URL = "https://api.dexscreener.com"
    
    def __init__(self, rate_limit: int = None):
        self.rate_limit = rate_limit or SETTINGS["dexscreener_rate_limit"]
        self._limiter = RateLimiter(max_requests=self.rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, endpoint: str, params: Dict[str, Any] = None, _retry_count: int = 0) -> Optional[Dict]:
        if _retry_count >= 3:
            logger.error(f"Max retries reached for {endpoint}")
            return None
        
        await self._limiter.acquire()
        session = await self._get_session()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    backoff = 30 * (2 ** _retry_count)  # 30s, 60s, 120s
                    logger.warning(f"Rate limited by DexScreener, waiting {backoff}s...")
                    await asyncio.sleep(backoff)
                    return await self._request(endpoint, params, _retry_count + 1)
                else:
                    logger.error(f"DexScreener API error: {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout requesting {endpoint}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {e}")
            return None
    
    async def get_latest_token_profiles(self) -> List[Dict[str, Any]]:
        """
        Obtiene los perfiles de tokens más recientes.
        Endpoint: GET /token-profiles/latest/v1
        """
        data = await self._request("/token-profiles/latest/v1")
        
        if not data:
            return []
        
        solana_tokens = [
            token for token in data 
            if token.get("chainId") == "solana"
        ]
        
        return solana_tokens
    
    async def get_latest_boosted_tokens(self) -> List[Dict[str, Any]]:
        """
        Obtiene tokens con boost activo (más visibilidad).
        Endpoint: GET /token-boosts/latest/v1
        """
        data = await self._request("/token-boosts/latest/v1")
        
        if not data:
            return []
        
        solana_tokens = [
            token for token in data 
            if token.get("chainId") == "solana"
        ]
        
        return solana_tokens
    
    async def get_token_pairs(self, token_address: str) -> List[Dict[str, Any]]:
        """
        Obtiene información detallada de pares para un token.
        Endpoint: GET /tokens/v1/solana/{tokenAddress}
        """
        data = await self._request(f"/tokens/v1/solana/{token_address}")
        
        if not data:
            return []
        
        return data if isinstance(data, list) else [data]
    
    async def get_pair_info(self, pair_address: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información detallada de un par específico.
        Endpoint: GET /pairs/solana/{pairAddress}
        """
        data = await self._request(f"/pairs/solana/{pair_address}")
        
        if data and "pair" in data:
            return data["pair"]
        elif data and "pairs" in data and data["pairs"]:
            return data["pairs"][0]
        
        return data
    
    async def search_pairs(self, query: str) -> List[Dict[str, Any]]:
        """
        Busca pares por nombre o dirección.
        Endpoint: GET /latest/dex/search?q={query}
        """
        data = await self._request("/latest/dex/search", {"q": query})
        
        if not data or "pairs" not in data:
            return []
        
        solana_pairs = [
            pair for pair in data["pairs"]
            if pair.get("chainId") == "solana"
        ]
        
        return solana_pairs
    
    async def get_new_solana_pairs(self, min_liquidity: float = None) -> List[Dict[str, Any]]:
        """
        Obtiene pares nuevos de Solana filtrando por liquidez.
        Usa el endpoint de búsqueda que devuelve datos de mercado reales.
        """
        min_liq = min_liquidity or SETTINGS["min_liquidity_usd"]
        all_pairs = []
        seen_addresses = set()
        
        profiles = await self.get_latest_token_profiles()
        
        for profile in profiles:
            token_address = profile.get("tokenAddress")
            if not token_address or token_address in seen_addresses:
                continue
            
            pairs = await self.get_token_pairs(token_address)
            
            if pairs:
                for pair in pairs:
                    liquidity = pair.get("liquidity", {}).get("usd", 0) or 0
                    if liquidity >= min_liq:
                        all_pairs.append(pair)
                        seen_addresses.add(token_address)
                        break
        
        return all_pairs
    
    async def get_trending_solana_pairs(self, min_liquidity: float = None) -> List[Dict[str, Any]]:
        """
        Obtiene pares trending de Solana usando búsqueda por términos populares.
        Alternativa cuando token-profiles no tiene suficientes datos.
        """
        min_liq = min_liquidity or SETTINGS["min_liquidity_usd"]
        all_pairs = []
        seen_addresses = set()
        
        search_terms = ["pump", "sol", "meme", "pepe", "doge", "cat", "ai", "trump"]
        
        for term in search_terms[:3]:
            pairs = await self.search_pairs(term)
            
            for pair in pairs:
                base_token = pair.get("baseToken", {})
                token_address = base_token.get("address")
                
                if not token_address or token_address in seen_addresses:
                    continue
                
                liquidity = pair.get("liquidity", {}).get("usd", 0) or 0
                pair_age = pair.get("pairCreatedAt")
                
                if pair_age:
                    try:
                        if isinstance(pair_age, (int, float)):
                            created = datetime.fromtimestamp(pair_age / 1000)
                        else:
                            created = datetime.fromisoformat(str(pair_age).replace("Z", "+00:00"))
                        
                        age_hours = (datetime.now() - created).total_seconds() / 3600
                        
                        if age_hours > 24:
                            continue
                    except:
                        pass
                
                if liquidity >= min_liq:
                    all_pairs.append(pair)
                    seen_addresses.add(token_address)
        
        return all_pairs
    
    def parse_token_data(self, pair_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parsea datos de par a formato interno normalizado.
        """
        base_token = pair_data.get("baseToken", {})
        quote_token = pair_data.get("quoteToken", {})
        liquidity = pair_data.get("liquidity", {})
        txns = pair_data.get("txns", {})
        volume = pair_data.get("volume", {})
        price_change = pair_data.get("priceChange", {})
        
        h24_txns = txns.get("h24", {})
        h1_txns = txns.get("h1", {})
        
        created_at = pair_data.get("pairCreatedAt")
        if created_at:
            if isinstance(created_at, (int, float)):
                created_at = datetime.fromtimestamp(created_at / 1000)
            else:
                created_at = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        
        age_hours = None
        if created_at:
            age_hours = (datetime.now() - created_at).total_seconds() / 3600
        
        buys_24h = h24_txns.get("buys", 0) or 0
        sells_24h = h24_txns.get("sells", 0) or 0
        buy_sell_ratio = buys_24h / max(sells_24h, 1)
        
        return {
            "address": base_token.get("address"),
            "symbol": base_token.get("symbol"),
            "name": base_token.get("name"),
            "pair_address": pair_data.get("pairAddress"),
            "dex": pair_data.get("dexId"),
            "created_at": created_at.isoformat() if created_at else None,
            "price_usd": float(pair_data.get("priceUsd", 0) or 0),
            "liquidity_usd": float(liquidity.get("usd", 0) or 0),
            "market_cap": float(pair_data.get("marketCap", 0) or 0),
            "fdv": float(pair_data.get("fdv", 0) or 0),
            "volume_24h": float(volume.get("h24", 0) or 0),
            "volume_1h": float(volume.get("h1", 0) or 0),
            "price_change_1h": float(price_change.get("h1", 0) or 0),
            "price_change_24h": float(price_change.get("h24", 0) or 0),
            "tx_count_24h": buys_24h + sells_24h,
            "tx_count_1h": (h1_txns.get("buys", 0) or 0) + (h1_txns.get("sells", 0) or 0),
            "buys_24h": buys_24h,
            "sells_24h": sells_24h,
            "buy_sell_ratio": buy_sell_ratio,
            "age_hours": age_hours,
            "quote_token": quote_token.get("symbol"),
            "ownership_renounced": None,
            "contract_verified": None,
            "_raw": pair_data,
        }
    
    async def get_prices_batch(self, token_addresses: List[str]) -> Dict[str, float]:
        """
        Obtiene precios actuales para múltiples tokens.
        """
        prices = {}
        
        for address in token_addresses:
            pairs = await self.get_token_pairs(address)
            if pairs:
                best_pair = max(pairs, key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0)
                prices[address] = float(best_pair.get("priceUsd", 0) or 0)
            else:
                prices[address] = 0.0
        
        return prices
    
    async def monitor_token_price(
        self, 
        token_address: str, 
        callback: callable,
        interval_seconds: int = None
    ) -> None:
        """
        Monitorea precio de un token y llama callback con cada actualización.
        """
        interval = interval_seconds or SETTINGS["price_poll_seconds"]
        
        while True:
            pairs = await self.get_token_pairs(token_address)
            
            if pairs:
                best_pair = max(pairs, key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0)
                price_data = {
                    "token_address": token_address,
                    "timestamp": datetime.now(),
                    "price_usd": float(best_pair.get("priceUsd", 0) or 0),
                    "volume": float(best_pair.get("volume", {}).get("h1", 0) or 0),
                    "liquidity": float(best_pair.get("liquidity", {}).get("usd", 0) or 0),
                }
                await callback(price_data)
            
            await asyncio.sleep(interval)
