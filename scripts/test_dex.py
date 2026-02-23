"""Test DexScreener client."""
import asyncio
import sys
sys.path.insert(0, '.')
from src.data_sources import DexScreenerClient

async def test():
    client = DexScreenerClient()
    
    print("Testing get_new_solana_pairs...")
    pairs = await client.get_new_solana_pairs(min_liquidity=5000)
    print(f"  Result: {len(pairs)} pairs")
    
    if len(pairs) < 5:
        print("\nTesting get_trending_solana_pairs...")
        trending = await client.get_trending_solana_pairs(min_liquidity=5000)
        print(f"  Result: {len(trending)} pairs")
        pairs.extend(trending)
    
    await client.close()
    
    print(f"\nTotal pairs found: {len(pairs)}")
    
    for i, p in enumerate(pairs[:5]):
        base = p.get("baseToken", {})
        liq = p.get("liquidity", {}).get("usd", 0) or 0
        price = p.get("priceUsd", 0)
        print(f"  {i+1}. {base.get('symbol', '??')} - Liq: ${liq:,.0f} - Price: ${float(price):.8f}")

if __name__ == "__main__":
    asyncio.run(test())
