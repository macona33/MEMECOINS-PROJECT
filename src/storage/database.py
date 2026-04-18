"""
SQLite database manager para estado de tokens, trades y métricas.
"""

import aiosqlite
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from loguru import logger

from config.settings import DATABASE_PATH, DATA_DIR


class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info(f"Connected to database: {self.db_path}")
    
    async def close(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    async def _create_tables(self) -> None:
        await self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS tokens (
                address TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                created_at TIMESTAMP,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                liquidity_usd REAL,
                ownership_renounced BOOLEAN,
                contract_verified BOOLEAN,
                dex TEXT,
                pair_address TEXT,
                price_usd REAL,
                market_cap REAL
            );
            
            CREATE TABLE IF NOT EXISTS token_features (
                address TEXT PRIMARY KEY,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                holder_concentration REAL,
                top_10_holders_pct REAL,
                liquidity_depth REAL,
                volume_24h REAL,
                volume_1h REAL,
                price_change_1h REAL,
                price_change_24h REAL,
                price_volatility REAL,
                age_hours REAL,
                tx_count_1h INTEGER,
                buy_sell_ratio REAL,
                FOREIGN KEY (address) REFERENCES tokens(address)
            );
            
            CREATE TABLE IF NOT EXISTS token_scores (
                address TEXT PRIMARY KEY,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                p_rug REAL,
                p_pump REAL,
                expected_g REAL,
                evs REAL,
                evs_adj REAL,
                sigma_token REAL,
                rank INTEGER,
                FOREIGN KEY (address) REFERENCES tokens(address)
            );
            
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price REAL,
                exit_price REAL,
                position_size_usd REAL,
                kelly_fraction REAL,
                mfe REAL,
                mae REAL,
                pnl_pct REAL,
                pnl_usd REAL,
                label TEXT,
                exit_reason TEXT,
                duration_minutes REAL,
                evs_at_entry REAL,
                p_rug_at_entry REAL,
                p_pump_at_entry REAL,
                FOREIGN KEY (token_address) REFERENCES tokens(address)
            );
            
            CREATE TABLE IF NOT EXISTS active_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT UNIQUE,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entry_price REAL,
                position_size_usd REAL,
                kelly_fraction REAL,
                current_price REAL,
                current_mfe REAL DEFAULT 0,
                current_mae REAL DEFAULT 0,
                stop_price REAL,
                take_profit_price REAL,
                FOREIGN KEY (token_address) REFERENCES tokens(address)
            );
            
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date DATE PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                hit_rate REAL,
                avg_evs_adj REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                total_pnl_pct REAL,
                total_pnl_usd REAL,
                avg_hold_duration REAL,
                pump_count INTEGER DEFAULT 0,
                rug_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS model_params (
                model_name TEXT,
                param_name TEXT,
                param_value REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_name, param_name)
            );
            
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                price_usd REAL,
                volume REAL,
                liquidity REAL,
                FOREIGN KEY (token_address) REFERENCES tokens(address)
            );
            
            -- v2.0: Features completas por trade para recalibración
            CREATE TABLE IF NOT EXISTS trade_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                token_address TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- Features estructurales
                holder_concentration REAL,
                top_holder_pct REAL,
                liquidity_ratio REAL,
                age_hours REAL,
                has_renounced INTEGER,
                has_verified INTEGER,
                has_freeze_auth INTEGER,
                has_mint_auth INTEGER,
                volume_liquidity_ratio REAL,
                structural_risk REAL,
                -- Features dinámicas
                volume_momentum REAL,
                price_momentum REAL,
                tx_momentum REAL,
                buy_pressure REAL,
                liquidity_quality REAL,
                opportunity_score REAL,
                log_liquidity REAL,
                volatility_24h REAL,
                momentum_score REAL,
                -- Scores al momento de entrada
                p_rug_predicted REAL,
                p_pump_predicted REAL,
                estimated_g REAL,
                evs REAL,
                evs_adj REAL,
                sigma_at_entry REAL,
                -- Stops configurados
                stop_loss_pct REAL,
                take_profit_pct REAL,
                -- Outcome real
                actual_mfe REAL,
                actual_mae REAL,
                actual_pnl REAL,
                stop_executed INTEGER,
                label TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades(id),
                FOREIGN KEY (token_address) REFERENCES tokens(address)
            );
            
            -- v2.0: Estado de riesgo para control de drawdown
            CREATE TABLE IF NOT EXISTS risk_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_drawdown REAL DEFAULT 0,
                peak_equity REAL DEFAULT 10000,
                current_equity REAL DEFAULT 10000,
                consecutive_losses INTEGER DEFAULT 0,
                frozen_until TIMESTAMP,
                gamma_multiplier REAL DEFAULT 1.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- v2.0: Historial de G por bucket de EVS
            CREATE TABLE IF NOT EXISTS g_historical (
                bucket_id TEXT PRIMARY KEY,
                evs_min REAL,
                evs_max REAL,
                avg_mfe REAL,
                sample_count INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- v2.0: Régimen de mercado
            CREATE TABLE IF NOT EXISTS market_regime (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                regime TEXT DEFAULT 'NORMAL',
                new_tokens_2h INTEGER DEFAULT 0,
                total_volume_2h REAL DEFAULT 0,
                pct_with_volume REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- v2.0: Estado de última recalibración (trigger: 30 trade_features nuevas desde aquí)
            CREATE TABLE IF NOT EXISTS calibration_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_recalibration_at TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_mint TEXT NOT NULL,
                amount_in_lamports INTEGER,
                expected_out_raw INTEGER,
                real_out_raw INTEGER,
                slippage_expected_bps INTEGER,
                slippage_real REAL,
                price_impact_pct REAL,
                tx_signature TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                execution_time_ms INTEGER,
                mode TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_tokens_detected ON tokens(detected_at);
            CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_price_snapshots ON price_snapshots(token_address, timestamp);
            CREATE INDEX IF NOT EXISTS idx_trade_features_trade ON trade_features(trade_id);
            CREATE INDEX IF NOT EXISTS idx_execution_logs_created ON execution_logs(created_at);
        """)
        await self._connection.commit()
    
    # ============== TOKEN OPERATIONS ==============
    
    async def insert_token(self, token: Dict[str, Any]) -> None:
        await self._connection.execute("""
            INSERT OR REPLACE INTO tokens 
            (address, symbol, name, created_at, liquidity_usd, ownership_renounced, 
             contract_verified, dex, pair_address, price_usd, market_cap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            token["address"],
            token.get("symbol"),
            token.get("name"),
            token.get("created_at"),
            token.get("liquidity_usd"),
            token.get("ownership_renounced"),
            token.get("contract_verified"),
            token.get("dex"),
            token.get("pair_address"),
            token.get("price_usd"),
            token.get("market_cap"),
        ))
        await self._connection.commit()
    
    async def get_token(self, address: str) -> Optional[Dict[str, Any]]:
        cursor = await self._connection.execute(
            "SELECT * FROM tokens WHERE address = ?", (address,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    
    async def get_recent_tokens(self, hours: int = 24) -> List[Dict[str, Any]]:
        cursor = await self._connection.execute("""
            SELECT * FROM tokens 
            WHERE detected_at > datetime('now', ?)
            ORDER BY detected_at DESC
        """, (f"-{hours} hours",))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    async def token_exists(self, address: str) -> bool:
        cursor = await self._connection.execute(
            "SELECT 1 FROM tokens WHERE address = ?", (address,)
        )
        return await cursor.fetchone() is not None
    
    # ============== FEATURES OPERATIONS ==============
    
    async def insert_features(self, address: str, features: Dict[str, Any]) -> None:
        await self._connection.execute("""
            INSERT OR REPLACE INTO token_features 
            (address, holder_concentration, top_10_holders_pct, liquidity_depth,
             volume_24h, volume_1h, price_change_1h, price_change_24h, 
             price_volatility, age_hours, tx_count_1h, buy_sell_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            address,
            features.get("holder_concentration"),
            features.get("top_10_holders_pct"),
            features.get("liquidity_depth"),
            features.get("volume_24h"),
            features.get("volume_1h"),
            features.get("price_change_1h"),
            features.get("price_change_24h"),
            features.get("price_volatility"),
            features.get("age_hours"),
            features.get("tx_count_1h"),
            features.get("buy_sell_ratio"),
        ))
        await self._connection.commit()
    
    async def get_features(self, address: str) -> Optional[Dict[str, Any]]:
        cursor = await self._connection.execute(
            "SELECT * FROM token_features WHERE address = ?", (address,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    
    # ============== SCORES OPERATIONS ==============
    
    async def insert_scores(self, address: str, scores: Dict[str, Any]) -> None:
        await self._connection.execute("""
            INSERT OR REPLACE INTO token_scores 
            (address, p_rug, p_pump, expected_g, evs, evs_adj, sigma_token, rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            address,
            scores.get("p_rug"),
            scores.get("p_pump"),
            scores.get("expected_g"),
            scores.get("evs"),
            scores.get("evs_adj"),
            scores.get("sigma_token"),
            scores.get("rank"),
        ))
        await self._connection.commit()
    
    async def get_scores(self, address: str) -> Optional[Dict[str, Any]]:
        cursor = await self._connection.execute(
            "SELECT * FROM token_scores WHERE address = ?", (address,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    
    async def get_top_ranked_tokens(self, limit: int = 10) -> List[Dict[str, Any]]:
        cursor = await self._connection.execute("""
            SELECT t.*, s.p_rug, s.p_pump, s.evs_adj, s.rank
            FROM tokens t
            JOIN token_scores s ON t.address = s.address
            WHERE s.evs_adj > 0
            ORDER BY s.evs_adj DESC
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ============== TRADE OPERATIONS ==============
    
    async def open_trade(self, trade: Dict[str, Any]) -> int:
        """
        Abre trade: inserta en trades (exit=NULL) y en active_trades.
        Retorna trades.id para que trade_features pueda guardarse al abrir.
        """
        # 1. Insertar en trades (exit fields NULL) - debe existir para trade_features
        cursor = await self._connection.execute("""
            INSERT INTO trades 
            (token_address, entry_time, entry_price, position_size_usd, kelly_fraction,
             evs_at_entry, p_rug_at_entry, p_pump_at_entry)
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?)
        """, (
            trade["token_address"],
            trade["entry_price"],
            trade["position_size_usd"],
            trade["kelly_fraction"],
            trade.get("evs_at_entry"),
            trade.get("p_rug_at_entry"),
            trade.get("p_pump_at_entry"),
        ))
        trade_id = cursor.lastrowid
        # 2. Insertar en active_trades
        await self._connection.execute("""
            INSERT INTO active_trades 
            (token_address, entry_price, position_size_usd, kelly_fraction, 
             current_price, stop_price, take_profit_price)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["token_address"],
            trade["entry_price"],
            trade["position_size_usd"],
            trade["kelly_fraction"],
            trade["entry_price"],
            trade["stop_price"],
            trade["take_profit_price"],
        ))
        await self._connection.commit()
        return trade_id
    
    async def get_active_trades(self) -> List[Dict[str, Any]]:
        cursor = await self._connection.execute("SELECT * FROM active_trades")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    async def update_active_trade(self, token_address: str, updates: Dict[str, Any]) -> None:
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [token_address]
        await self._connection.execute(
            f"UPDATE active_trades SET {set_clause} WHERE token_address = ?",
            values
        )
        await self._connection.commit()
    
    async def close_trade(self, token_address: str, result: Dict[str, Any]) -> None:
        active = await self._connection.execute(
            "SELECT * FROM active_trades WHERE token_address = ?", (token_address,)
        )
        active_trade = await active.fetchone()
        
        if active_trade:
            active_trade = dict(active_trade)
            # Actualizar fila en trades si existe (creada al abrir); si no, INSERT (compatibilidad)
            cursor = await self._connection.execute("""
                SELECT id FROM trades 
                WHERE token_address = ? AND exit_time IS NULL 
                ORDER BY id DESC LIMIT 1
            """, (token_address,))
            row = await cursor.fetchone()
            if row:
                await self._connection.execute("""
                    UPDATE trades SET
                        exit_time = ?,
                        exit_price = ?,
                        mfe = ?,
                        mae = ?,
                        pnl_pct = ?,
                        pnl_usd = ?,
                        label = ?,
                        exit_reason = ?,
                        duration_minutes = ?
                    WHERE id = ?
                """, (
                    datetime.now().isoformat(),
                    result["exit_price"],
                    result.get("mfe", active_trade.get("current_mfe", 0)),
                    result.get("mae", active_trade.get("current_mae", 0)),
                    result["pnl_pct"],
                    result["pnl_usd"],
                    result["label"],
                    result["exit_reason"],
                    result.get("duration_minutes"),
                    row[0],
                ))
            else:
                # Fallback: trades abiertos con código antiguo (solo active_trades)
                await self._connection.execute("""
                    INSERT INTO trades 
                    (token_address, entry_time, exit_time, entry_price, exit_price,
                     position_size_usd, kelly_fraction, mfe, mae, pnl_pct, pnl_usd,
                     label, exit_reason, duration_minutes, evs_at_entry, p_rug_at_entry, p_pump_at_entry)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token_address,
                    active_trade["entry_time"],
                    datetime.now().isoformat(),
                    active_trade["entry_price"],
                    result["exit_price"],
                    active_trade["position_size_usd"],
                    active_trade["kelly_fraction"],
                    result.get("mfe", active_trade.get("current_mfe", 0)),
                    result.get("mae", active_trade.get("current_mae", 0)),
                    result["pnl_pct"],
                    result["pnl_usd"],
                    result["label"],
                    result["exit_reason"],
                    result.get("duration_minutes"),
                    result.get("evs_at_entry"),
                    result.get("p_rug_at_entry"),
                    result.get("p_pump_at_entry"),
                ))
            
            await self._connection.execute(
                "DELETE FROM active_trades WHERE token_address = ?", (token_address,)
            )
            await self._connection.commit()
    
    async def get_trade_history(self, days: int = 30) -> List[Dict[str, Any]]:
        cursor = await self._connection.execute("""
            SELECT * FROM trades 
            WHERE entry_time > datetime('now', ?)
            ORDER BY entry_time DESC
        """, (f"-{days} days",))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ============== METRICS OPERATIONS ==============
    
    async def update_daily_metrics(self, date: str, metrics: Dict[str, Any]) -> None:
        await self._connection.execute("""
            INSERT OR REPLACE INTO daily_metrics 
            (date, total_trades, winning_trades, hit_rate, avg_evs_adj, 
             max_drawdown, sharpe_ratio, total_pnl_pct, total_pnl_usd,
             avg_hold_duration, pump_count, rug_count, neutral_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            metrics.get("total_trades", 0),
            metrics.get("winning_trades", 0),
            metrics.get("hit_rate"),
            metrics.get("avg_evs_adj"),
            metrics.get("max_drawdown"),
            metrics.get("sharpe_ratio"),
            metrics.get("total_pnl_pct"),
            metrics.get("total_pnl_usd"),
            metrics.get("avg_hold_duration"),
            metrics.get("pump_count", 0),
            metrics.get("rug_count", 0),
            metrics.get("neutral_count", 0),
        ))
        await self._connection.commit()
    
    async def get_daily_metrics(self, days: int = 30) -> List[Dict[str, Any]]:
        cursor = await self._connection.execute("""
            SELECT * FROM daily_metrics 
            WHERE date > date('now', ?)
            ORDER BY date DESC
        """, (f"-{days} days",))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ============== MODEL PARAMS OPERATIONS ==============
    
    async def save_model_params(self, model_name: str, params: Dict[str, float]) -> None:
        for param_name, param_value in params.items():
            await self._connection.execute("""
                INSERT OR REPLACE INTO model_params (model_name, param_name, param_value)
                VALUES (?, ?, ?)
            """, (model_name, param_name, param_value))
        await self._connection.commit()
    
    async def get_model_params(self, model_name: str) -> Dict[str, float]:
        cursor = await self._connection.execute(
            "SELECT param_name, param_value FROM model_params WHERE model_name = ?",
            (model_name,)
        )
        rows = await cursor.fetchall()
        return {row["param_name"]: row["param_value"] for row in rows}
    
    # ============== PRICE SNAPSHOTS ==============
    
    async def save_price_snapshot(self, token_address: str, price: float, 
                                   volume: float = None, liquidity: float = None) -> None:
        await self._connection.execute("""
            INSERT INTO price_snapshots (token_address, price_usd, volume, liquidity)
            VALUES (?, ?, ?, ?)
        """, (token_address, price, volume, liquidity))
        await self._connection.commit()
    
    async def get_price_history(self, token_address: str, hours: int = 24) -> List[Dict[str, Any]]:
        cursor = await self._connection.execute("""
            SELECT * FROM price_snapshots 
            WHERE token_address = ? AND timestamp > datetime('now', ?)
            ORDER BY timestamp ASC
        """, (token_address, f"-{hours} hours"))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ============== v2.0: TRADE FEATURES ==============
    
    async def save_trade_features(self, trade_id: int, token_address: str, 
                                   features: Dict[str, Any], scores: Dict[str, Any],
                                   stops: Dict[str, float]) -> int:
        """Guarda snapshot completo de features para recalibración."""
        cursor = await self._connection.execute("""
            INSERT INTO trade_features 
            (trade_id, token_address, holder_concentration, top_holder_pct, liquidity_ratio,
             age_hours, has_renounced, has_verified, has_freeze_auth, has_mint_auth,
             volume_liquidity_ratio, structural_risk, volume_momentum, price_momentum,
             tx_momentum, buy_pressure, liquidity_quality, opportunity_score, log_liquidity,
             volatility_24h, momentum_score, p_rug_predicted, p_pump_predicted, estimated_g,
             evs, evs_adj, sigma_at_entry, stop_loss_pct, take_profit_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            token_address,
            features.get("holder_concentration"),
            features.get("top_holder_pct"),
            features.get("liquidity_ratio"),
            features.get("age_hours"),
            int(features.get("has_renounced", 0)),
            int(features.get("has_verified", 0)),
            int(features.get("has_freeze_auth", 0)),
            int(features.get("has_mint_auth", 0)),
            features.get("volume_liquidity_ratio"),
            features.get("structural_risk"),
            features.get("volume_momentum"),
            features.get("price_momentum"),
            features.get("tx_momentum"),
            features.get("buy_pressure"),
            features.get("liquidity_quality"),
            features.get("opportunity_score"),
            features.get("log_liquidity"),
            features.get("volatility_24h"),
            features.get("momentum_score"),
            scores.get("p_rug"),
            scores.get("p_pump"),
            scores.get("expected_g"),
            scores.get("evs"),
            scores.get("evs_adj"),
            scores.get("sigma"),
            stops.get("stop_loss_pct"),
            stops.get("take_profit_pct"),
        ))
        await self._connection.commit()
        return cursor.lastrowid
    
    async def update_trade_features_outcome(self, trade_id: int, 
                                             mfe: float, mae: float, pnl: float,
                                             stop_executed: bool, label: str) -> None:
        """Actualiza el outcome real de un trade para recalibración."""
        await self._connection.execute("""
            UPDATE trade_features 
            SET actual_mfe = ?, actual_mae = ?, actual_pnl = ?, 
                stop_executed = ?, label = ?
            WHERE trade_id = ?
        """, (mfe, mae, pnl, int(stop_executed), label, trade_id))
        await self._connection.commit()
    
    async def get_training_dataset(self, days: Optional[int] = 30) -> List[Dict[str, Any]]:
        """
        Obtiene dataset completo para recalibración.
        days=None: todas las trade_features con label (sin filtro de tiempo).
        days=N: solo las de los últimos N días.
        """
        if days is None:
            cursor = await self._connection.execute("""
                SELECT tf.*, t.exit_reason, t.duration_minutes
                FROM trade_features tf
                JOIN trades t ON tf.trade_id = t.id
                WHERE tf.label IS NOT NULL
                ORDER BY tf.recorded_at DESC
            """)
        else:
            cursor = await self._connection.execute("""
                SELECT tf.*, t.exit_reason, t.duration_minutes
                FROM trade_features tf
                JOIN trades t ON tf.trade_id = t.id
                WHERE tf.recorded_at > datetime('now', ?)
                AND tf.label IS NOT NULL
                ORDER BY tf.recorded_at DESC
            """, (f"-{days} days",))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    async def get_last_recalibration_at(self) -> Optional[str]:
        """Fecha/hora de la última recalibración (o None si nunca)."""
        cursor = await self._connection.execute(
            "SELECT last_recalibration_at FROM calibration_state WHERE id = 1"
        )
        row = await cursor.fetchone()
        if row and row[0]:
            return row[0]
        return None
    
    async def set_last_recalibration_at(self, at: str = None) -> None:
        """Registra que se acaba de ejecutar una recalibración."""
        at = at or datetime.now().isoformat()
        await self._connection.execute("""
            INSERT OR REPLACE INTO calibration_state (id, last_recalibration_at, last_updated)
            VALUES (1, ?, CURRENT_TIMESTAMP)
        """, (at,))
        await self._connection.commit()
    
    async def count_trade_features_since(self, since_iso: Optional[str] = None) -> int:
        """
        Cuenta trade_features con label.
        since_iso=None: total con label.
        since_iso=str: las que tienen recorded_at > since_iso (nuevas desde última recal).
        """
        if since_iso is None:
            cursor = await self._connection.execute(
                "SELECT COUNT(*) FROM trade_features WHERE label IS NOT NULL"
            )
        else:
            cursor = await self._connection.execute(
                "SELECT COUNT(*) FROM trade_features WHERE label IS NOT NULL AND recorded_at > ?",
                (since_iso,)
            )
        row = await cursor.fetchone()
        return row[0] if row else 0
    
    async def get_trade_count(self) -> int:
        """Retorna número total de trades con features guardadas."""
        cursor = await self._connection.execute(
            "SELECT COUNT(*) FROM trade_features WHERE label IS NOT NULL"
        )
        row = await cursor.fetchone()
        return row[0] if row else 0
    
    # ============== v2.0: RISK STATE ==============
    
    async def get_risk_state(self) -> Dict[str, Any]:
        """Obtiene estado actual de riesgo."""
        cursor = await self._connection.execute("SELECT * FROM risk_state WHERE id = 1")
        row = await cursor.fetchone()
        if row:
            return dict(row)
        await self._connection.execute("""
            INSERT OR IGNORE INTO risk_state (id) VALUES (1)
        """)
        await self._connection.commit()
        return {
            "current_drawdown": 0,
            "peak_equity": 10000,
            "current_equity": 10000,
            "consecutive_losses": 0,
            "frozen_until": None,
            "gamma_multiplier": 1.0,
        }
    
    async def update_risk_state(self, updates: Dict[str, Any]) -> None:
        """Actualiza estado de riesgo."""
        updates["last_updated"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values())
        await self._connection.execute(
            f"UPDATE risk_state SET {set_clause} WHERE id = 1", values
        )
        await self._connection.commit()
    
    # ============== v2.0: G HISTORICAL ==============
    
    async def update_g_bucket(self, bucket_id: str, evs_min: float, evs_max: float,
                               avg_mfe: float, sample_count: int) -> None:
        """Actualiza promedio de MFE por bucket de EVS."""
        await self._connection.execute("""
            INSERT OR REPLACE INTO g_historical 
            (bucket_id, evs_min, evs_max, avg_mfe, sample_count, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (bucket_id, evs_min, evs_max, avg_mfe, sample_count))
        await self._connection.commit()
    
    async def get_g_buckets(self) -> List[Dict[str, Any]]:
        """Obtiene todos los buckets de G histórico."""
        cursor = await self._connection.execute(
            "SELECT * FROM g_historical ORDER BY evs_min"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ============== v2.0: MARKET REGIME ==============
    
    async def get_market_regime(self) -> Dict[str, Any]:
        """Obtiene régimen de mercado actual."""
        cursor = await self._connection.execute("SELECT * FROM market_regime WHERE id = 1")
        row = await cursor.fetchone()
        if row:
            return dict(row)
        await self._connection.execute("""
            INSERT OR IGNORE INTO market_regime (id) VALUES (1)
        """)
        await self._connection.commit()
        return {"regime": "NORMAL", "new_tokens_2h": 0, "total_volume_2h": 0, "pct_with_volume": 0}
    
    async def update_market_regime(self, regime: str, metrics: Dict[str, Any]) -> None:
        """Actualiza régimen de mercado."""
        await self._connection.execute("""
            UPDATE market_regime 
            SET regime = ?, new_tokens_2h = ?, total_volume_2h = ?, 
                pct_with_volume = ?, last_updated = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (
            regime, 
            metrics.get("new_tokens_2h", 0),
            metrics.get("total_volume_2h", 0),
            metrics.get("pct_with_volume", 0),
        ))
        await self._connection.commit()
    
    async def get_last_trade_id(self, token_address: str) -> Optional[int]:
        """Obtiene el ID del último trade para un token."""
        cursor = await self._connection.execute("""
            SELECT id FROM trades 
            WHERE token_address = ? 
            ORDER BY id DESC LIMIT 1
        """, (token_address,))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def count_closed_trades_for_token(self, token_address: str) -> int:
        """Cuántos trades ya cerrados existen para este mint (re-entrada controlada)."""
        cursor = await self._connection.execute(
            """
            SELECT COUNT(*) FROM trades
            WHERE token_address = ?
              AND exit_time IS NOT NULL
            """,
            (token_address,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # ============== FASE 3: EXECUTION LOGS (Jupiter) ==============

    async def insert_execution_log(self, row: Dict[str, Any]) -> int:
        """Inserta un registro de ejecución on-chain (quote/swap)."""
        cursor = await self._connection.execute("""
            INSERT INTO execution_logs (
                token_mint, amount_in_lamports, expected_out_raw, real_out_raw,
                slippage_expected_bps, slippage_real, price_impact_pct,
                tx_signature, status, error_message, execution_time_ms, mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("token_mint"),
            row.get("amount_in_lamports"),
            row.get("expected_out_raw"),
            row.get("real_out_raw"),
            row.get("slippage_expected_bps"),
            row.get("slippage_real"),
            row.get("price_impact_pct"),
            row.get("tx_signature"),
            row.get("status"),
            row.get("error_message"),
            row.get("execution_time_ms"),
            row.get("mode"),
        ))
        await self._connection.commit()
        return cursor.lastrowid
