"""
Parquet manager para series temporales de precios.
Optimizado para análisis con pandas y recalibración de modelos.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger

from config.settings import TIMESERIES_DIR


class TimeseriesManager:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or TIMESERIES_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._schema = pa.schema([
            ("token_address", pa.string()),
            ("timestamp", pa.timestamp("ms")),
            ("price_usd", pa.float64()),
            ("volume", pa.float64()),
            ("liquidity", pa.float64()),
            ("market_cap", pa.float64()),
        ])
    
    def _get_partition_path(self, date: datetime) -> Path:
        return self.base_dir / f"prices_{date.strftime('%Y%m%d')}.parquet"
    
    async def append_prices(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        for date, group in df.groupby(df["timestamp"].dt.date):
            partition_path = self._get_partition_path(datetime.combine(date, datetime.min.time()))
            
            if partition_path.exists():
                existing = pd.read_parquet(partition_path)
                combined = pd.concat([existing, group], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["token_address", "timestamp"], 
                    keep="last"
                )
            else:
                combined = group
            
            combined.to_parquet(partition_path, index=False)
        
        logger.debug(f"Appended {len(records)} price records")
    
    async def get_token_prices(
        self, 
        token_address: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)
        
        all_data = []
        current = start_date
        
        while current <= end_date:
            partition_path = self._get_partition_path(current)
            if partition_path.exists():
                df = pd.read_parquet(partition_path)
                token_data = df[df["token_address"] == token_address]
                if not token_data.empty:
                    all_data.append(token_data)
            current += timedelta(days=1)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values("timestamp")
            return result
        
        return pd.DataFrame(columns=self._schema.names)
    
    async def get_all_prices(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        token_addresses: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)
        
        all_data = []
        current = start_date
        
        while current <= end_date:
            partition_path = self._get_partition_path(current)
            if partition_path.exists():
                df = pd.read_parquet(partition_path)
                if token_addresses:
                    df = df[df["token_address"].isin(token_addresses)]
                if not df.empty:
                    all_data.append(df)
            current += timedelta(days=1)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values(["token_address", "timestamp"])
            return result
        
        return pd.DataFrame(columns=self._schema.names)
    
    async def calculate_volatility(
        self, 
        token_address: str, 
        window_hours: int = 24
    ) -> Optional[float]:
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=window_hours)
        
        df = await self.get_token_prices(token_address, start_date, end_date)
        
        if len(df) < 10:
            return None
        
        df = df.sort_values("timestamp")
        df["returns"] = df["price_usd"].pct_change()
        
        volatility = df["returns"].std()
        return volatility if pd.notna(volatility) else None
    
    async def calculate_mfe_mae(
        self,
        token_address: str,
        entry_time: datetime,
        entry_price: float,
        end_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        if end_time is None:
            end_time = datetime.now()
        
        df = await self.get_token_prices(token_address, entry_time, end_time)
        
        if df.empty:
            return {"mfe": 0.0, "mae": 0.0, "mfe_time": None, "mae_time": None}
        
        df["pct_change"] = (df["price_usd"] - entry_price) / entry_price
        
        mfe = df["pct_change"].max()
        mae = df["pct_change"].min()
        
        mfe_idx = df["pct_change"].idxmax()
        mae_idx = df["pct_change"].idxmin()
        
        return {
            "mfe": mfe,
            "mae": mae,
            "mfe_time": df.loc[mfe_idx, "timestamp"] if pd.notna(mfe_idx) else None,
            "mae_time": df.loc[mae_idx, "timestamp"] if pd.notna(mae_idx) else None,
        }
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        removed = 0
        
        for file in self.base_dir.glob("prices_*.parquet"):
            try:
                date_str = file.stem.replace("prices_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                if file_date < cutoff:
                    file.unlink()
                    removed += 1
            except (ValueError, OSError) as e:
                logger.warning(f"Could not process file {file}: {e}")
        
        if removed:
            logger.info(f"Cleaned up {removed} old parquet files")
        
        return removed
    
    async def get_training_dataset(
        self,
        days: int = 30,
        min_datapoints: int = 100
    ) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = await self.get_all_prices(start_date, end_date)
        
        if df.empty:
            return df
        
        token_counts = df.groupby("token_address").size()
        valid_tokens = token_counts[token_counts >= min_datapoints].index
        
        return df[df["token_address"].isin(valid_tokens)]
