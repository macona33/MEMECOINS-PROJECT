"""
Cliente async para Solscan API.
Usado para verificar ownership renounced y datos de holders.
API gratuita con límites.
"""

import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from loguru import logger

from config.api_keys import API_KEYS


class SolscanClient:
    """Cliente para Solscan API - verificación de tokens."""
    
    BASE_URL = "https://pro-api.solscan.io/v2.0"
    PUBLIC_URL = "https://api.solscan.io"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEYS.get("solscan", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._last_request = 0
        self._min_interval = 0.5
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["token"] = self.api_key
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, endpoint: str, params: Dict[str, Any] = None, use_pro: bool = True) -> Optional[Dict]:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request = asyncio.get_event_loop().time()
        
        session = await self._get_session()
        base_url = self.BASE_URL if (use_pro and self.api_key) else self.PUBLIC_URL
        url = f"{base_url}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", data) if isinstance(data, dict) else data
                elif response.status == 429:
                    logger.warning("Solscan rate limited, backing off...")
                    await asyncio.sleep(5)
                    return None
                else:
                    logger.debug(f"Solscan API returned {response.status} for {endpoint}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout requesting Solscan {endpoint}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Solscan client error: {e}")
            return None
    
    async def get_token_meta(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene metadata del token.
        """
        data = await self._request(f"/token/meta", {"address": token_address})
        return data
    
    async def get_token_holders(
        self, 
        token_address: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Obtiene lista de top holders del token.
        """
        data = await self._request(
            f"/token/holders",
            {"address": token_address, "page": 1, "page_size": limit}
        )
        
        if data and "items" in data:
            return data["items"]
        elif isinstance(data, list):
            return data[:limit]
        
        return []
    
    async def get_token_transfer_stats(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene estadísticas de transferencias del token.
        """
        return await self._request(f"/token/transfer", {"address": token_address})
    
    async def check_ownership_renounced(self, token_address: str) -> Optional[bool]:
        """
        Verifica si el ownership del token ha sido renunciado.
        En Solana, esto se verifica mediante:
        1. Mint authority = null
        2. Freeze authority = null
        """
        meta = await self.get_token_meta(token_address)
        
        if not meta:
            return None
        
        mint_authority = meta.get("mintAuthority") or meta.get("mint_authority")
        freeze_authority = meta.get("freezeAuthority") or meta.get("freeze_authority")
        
        mint_renounced = mint_authority is None or mint_authority == ""
        freeze_renounced = freeze_authority is None or freeze_authority == ""
        
        return mint_renounced and freeze_renounced
    
    async def calculate_holder_concentration(self, token_address: str) -> Optional[Dict[str, float]]:
        """
        Calcula métricas de concentración de holders.
        """
        holders = await self.get_token_holders(token_address, limit=20)
        
        if not holders:
            return None
        
        total_supply = sum(
            float(h.get("amount", 0) or h.get("balance", 0) or 0) 
            for h in holders
        )
        
        if total_supply == 0:
            return None
        
        top_holder_pct = 0
        top_5_pct = 0
        top_10_pct = 0
        
        for i, holder in enumerate(holders):
            amount = float(holder.get("amount", 0) or holder.get("balance", 0) or 0)
            pct = amount / total_supply if total_supply > 0 else 0
            
            if i == 0:
                top_holder_pct = pct
            if i < 5:
                top_5_pct += pct
            if i < 10:
                top_10_pct += pct
        
        return {
            "top_holder_pct": top_holder_pct,
            "top_5_holders_pct": top_5_pct,
            "top_10_holders_pct": top_10_pct,
            "holder_count": len(holders),
        }
    
    async def get_token_verification_status(self, token_address: str) -> Dict[str, Any]:
        """
        Obtiene estado de verificación y seguridad del token.
        """
        meta = await self.get_token_meta(token_address)
        
        if not meta:
            return {
                "ownership_renounced": None,
                "contract_verified": None,
                "has_freeze_authority": None,
                "has_mint_authority": None,
            }
        
        mint_authority = meta.get("mintAuthority") or meta.get("mint_authority")
        freeze_authority = meta.get("freezeAuthority") or meta.get("freeze_authority")
        
        return {
            "ownership_renounced": (not mint_authority) and (not freeze_authority),
            "contract_verified": meta.get("verified", False),
            "has_freeze_authority": bool(freeze_authority),
            "has_mint_authority": bool(mint_authority),
            "decimals": meta.get("decimals"),
            "supply": meta.get("supply"),
            "symbol": meta.get("symbol"),
            "name": meta.get("name"),
        }
    
    async def enrich_token_data(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriquece datos del token con información de Solscan.
        """
        address = token_data.get("address")
        if not address:
            return token_data
        
        enriched = token_data.copy()
        
        verification = await self.get_token_verification_status(address)
        enriched.update({
            "ownership_renounced": verification.get("ownership_renounced"),
            "contract_verified": verification.get("contract_verified"),
            "has_freeze_authority": verification.get("has_freeze_authority"),
            "has_mint_authority": verification.get("has_mint_authority"),
        })
        
        concentration = await self.calculate_holder_concentration(address)
        if concentration:
            enriched.update({
                "top_holder_pct": concentration.get("top_holder_pct"),
                "top_5_holders_pct": concentration.get("top_5_holders_pct"),
                "top_10_holders_pct": concentration.get("top_10_holders_pct"),
                "holder_count": concentration.get("holder_count"),
            })
        
        return enriched
