"""Tests: imputación de booleanos on-chain desconocidos (Dex sin enrich)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.features import (
    UNKNOWN_BOOL_FEATURE,
    FeatureExtractor,
    _authority_or_absent,
    _bool_or_unknown,
)


def test_bool_or_unknown():
    assert _bool_or_unknown(None) == UNKNOWN_BOOL_FEATURE
    assert _bool_or_unknown(True) == 1.0
    assert _bool_or_unknown(False) == 0.0


def test_authority_or_absent():
    assert _authority_or_absent(None) == 0.0
    assert _authority_or_absent(True) == 1.0
    assert _authority_or_absent(False) == 0.0


def test_extract_static_dex_like_token():
    fx = FeatureExtractor(MagicMock())
    token = {
        "address": "So11111111111111111111111111111111111111112",
        "liquidity_usd": 15_000,
        "ownership_renounced": None,
        "contract_verified": None,
    }
    static = fx.extract_static_features(token)
    assert static["has_renounced"] == UNKNOWN_BOOL_FEATURE
    assert static["has_verified"] == UNKNOWN_BOOL_FEATURE
    assert static["has_freeze_auth"] == 0.0
    assert static["has_mint_auth"] == 0.0


def test_structural_risk_skips_unknown_renounced():
    fx = FeatureExtractor(MagicMock())
    base = {
        "has_freeze_auth": 0.0,
        "has_mint_auth": 0.0,
        "has_renounced": UNKNOWN_BOOL_FEATURE,
        "holder_concentration": 0.5,
        "liquidity_ratio": 0.1,
    }
    risk = fx.calculate_risk_features(base)
    # Antes: has_renounced==0 añadía 0.3; con desconocido no debe aplicar ese tramo.
    assert risk["structural_risk"] < 0.35


@pytest.mark.asyncio
async def test_full_pipeline_unknown_renounced_no_db_write_errors():
    fx = FeatureExtractor(MagicMock())
    token = {
        "address": "So11111111111111111111111111111111111111112",
        "liquidity_usd": 15_000,
        "market_cap": 50_000,
        "ownership_renounced": None,
        "contract_verified": None,
        "age_hours": 2.0,
        "volume_24h": 100_000,
        "volume_1h": 5_000,
    }
    feats = await fx.extract_all_features(token)
    assert feats["has_renounced"] == UNKNOWN_BOOL_FEATURE
