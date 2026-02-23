"""
Filtros de calidad para tokens.
Aplica reglas de filtrado para descartar tokens de alto riesgo.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from config.settings import SETTINGS


@dataclass
class FilterResult:
    """Resultado de aplicar filtros a un token."""
    passed: bool
    token: Dict[str, Any]
    failed_filters: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def rejection_reason(self) -> Optional[str]:
        if self.failed_filters:
            return ", ".join(self.failed_filters)
        return None


class TokenFilter:
    """Aplica filtros de calidad a tokens."""
    
    def __init__(self, settings: Dict[str, Any] = None):
        self.settings = settings or SETTINGS
    
    def filter_liquidity(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verifica liquidez mínima."""
        liquidity = token.get("liquidity_usd", 0) or 0
        min_liq = self.settings["min_liquidity_usd"]
        
        if liquidity < min_liq:
            return False, f"Low liquidity: ${liquidity:,.0f} < ${min_liq:,.0f}"
        
        return True, None
    
    def filter_ownership(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verifica si ownership está renunciado."""
        if not self.settings["require_renounced"]:
            return True, None
        
        renounced = token.get("ownership_renounced")
        
        if renounced is None:
            return True, None
        
        if not renounced:
            return False, "Ownership not renounced"
        
        return True, None
    
    def filter_verification(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verifica si contrato está verificado."""
        if not self.settings["require_verified"]:
            return True, None
        
        verified = token.get("contract_verified")
        
        if verified is None:
            return True, None
        
        if not verified:
            return False, "Contract not verified"
        
        return True, None
    
    def filter_holder_concentration(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verifica concentración de holders."""
        top_10_pct = token.get("top_10_holders_pct")
        max_concentration = self.settings["max_holder_concentration"]
        
        if top_10_pct is None:
            return True, None
        
        if top_10_pct > max_concentration:
            return False, f"High holder concentration: {top_10_pct:.1%} > {max_concentration:.1%}"
        
        return True, None
    
    def filter_age(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verifica edad del token."""
        age_hours = token.get("age_hours")
        
        if age_hours is None:
            return True, None
        
        min_age = self.settings["min_token_age_minutes"] / 60
        max_age = self.settings["max_token_age_hours"]
        
        if age_hours < min_age:
            return False, f"Too new: {age_hours*60:.0f}min < {min_age*60:.0f}min"
        
        if age_hours > max_age:
            return False, f"Too old: {age_hours:.1f}h > {max_age}h"
        
        return True, None
    
    def filter_freeze_authority(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verifica si token tiene freeze authority (riesgo)."""
        has_freeze = token.get("has_freeze_authority")
        
        if has_freeze is True:
            return True, "WARNING: Has freeze authority"
        
        return True, None
    
    def filter_mint_authority(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verifica si token tiene mint authority (riesgo de inflación)."""
        has_mint = token.get("has_mint_authority")
        
        if has_mint is True:
            return True, "WARNING: Has mint authority"
        
        return True, None
    
    def filter_volume(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verifica volumen mínimo."""
        volume_24h = token.get("volume_24h", 0) or 0
        liquidity = token.get("liquidity_usd", 0) or 0
        
        if liquidity > 0 and volume_24h < liquidity * 0.1:
            return True, "WARNING: Low volume relative to liquidity"
        
        return True, None
    
    def filter_suspicious_patterns(self, token: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Detecta patrones sospechosos."""
        warnings = []
        
        buy_sell_ratio = token.get("buy_sell_ratio", 1)
        if buy_sell_ratio > 10:
            warnings.append("Unusual buy/sell ratio")
        
        price_change_1h = token.get("price_change_1h", 0) or 0
        if price_change_1h > 500:
            warnings.append("Extreme 1h price increase")
        
        top_holder = token.get("top_holder_pct", 0) or 0
        if top_holder > 0.3:
            warnings.append(f"Single holder owns {top_holder:.1%}")
        
        if warnings:
            return True, "WARNING: " + "; ".join(warnings)
        
        return True, None
    
    def apply_filters(self, token: Dict[str, Any]) -> FilterResult:
        """
        Aplica todos los filtros a un token.
        Retorna FilterResult con resultado y detalles.
        """
        failed = []
        warnings = []
        
        filters = [
            ("liquidity", self.filter_liquidity),
            ("ownership", self.filter_ownership),
            ("verification", self.filter_verification),
            ("holder_concentration", self.filter_holder_concentration),
            ("age", self.filter_age),
            ("freeze_authority", self.filter_freeze_authority),
            ("mint_authority", self.filter_mint_authority),
            ("volume", self.filter_volume),
            ("suspicious_patterns", self.filter_suspicious_patterns),
        ]
        
        for name, filter_func in filters:
            passed, message = filter_func(token)
            
            if not passed:
                failed.append(message or name)
            elif message and message.startswith("WARNING"):
                warnings.append(message)
        
        result = FilterResult(
            passed=len(failed) == 0,
            token=token,
            failed_filters=failed,
            warnings=warnings,
        )
        
        if not result.passed:
            logger.debug(
                f"Token {token.get('symbol')} rejected: {result.rejection_reason}"
            )
        elif warnings:
            logger.debug(
                f"Token {token.get('symbol')} passed with warnings: {warnings}"
            )
        
        return result
    
    def filter_batch(self, tokens: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[FilterResult]]:
        """
        Filtra un lote de tokens.
        Retorna (tokens_aprobados, todos_los_resultados).
        """
        results = [self.apply_filters(t) for t in tokens]
        passed = [r.token for r in results if r.passed]
        
        logger.info(
            f"Filtered {len(tokens)} tokens: {len(passed)} passed, "
            f"{len(tokens) - len(passed)} rejected"
        )
        
        return passed, results
    
    def get_filter_stats(self, results: List[FilterResult]) -> Dict[str, int]:
        """
        Genera estadísticas de filtrado.
        """
        stats = {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "rejected": sum(1 for r in results if not r.passed),
            "with_warnings": sum(1 for r in results if r.warnings),
        }
        
        rejection_reasons = {}
        for r in results:
            for reason in r.failed_filters:
                key = reason.split(":")[0].strip()
                rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
        
        stats["rejection_reasons"] = rejection_reasons
        
        return stats
