"""
Recalibración v2.0 en standalone.
Solo recalibra cuando hay al menos 30 trade_features NUEVAS desde la última
recalibración (o 30 totales si es la primera vez). Entrena con todos los
datos disponibles (sin ventana fija de días).

Uso:
  python scripts/recalibrate_standalone.py
  python scripts/recalibrate_standalone.py --force   # ignora el mínimo de 30 nuevas
  python scripts/recalibrate_standalone.py --days 90 # entrena solo con últimos 90 días (opcional)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.storage import DatabaseManager
from src.models import HazardModel, PumpModel
from config.settings import SETTINGS


async def run_recalibration(force: bool = False, training_days: int = None) -> None:
    """
    Ejecuta recalibración si hay >= 30 trade_features nuevas desde la última.
    force=True: ejecuta igual (mínimo 30 totales).
    training_days: si se pasa, entrena solo con trade_features de los últimos N días; si None, usa todas.
    """
    min_new = SETTINGS.get("min_trades_for_recalibration", 30)

    db = DatabaseManager()
    await db.connect()

    try:
        last_at = await db.get_last_recalibration_at()
        count_new = await db.count_trade_features_since(last_at)
        count_total = await db.count_trade_features_since(None)

        print(f"\n[recalibrate_standalone]")
        print(f"  Ultima recalibracion: {last_at or 'nunca'}")
        print(f"  Trade features nuevas desde entonces: {count_new}")
        print(f"  Trade features totales con label: {count_total}")

        if not force:
            if count_total < min_new:
                print(f"  No se ejecuta: se necesitan al menos {min_new} trade_features (tienes {count_total}).")
                return
            if last_at is not None and count_new < min_new:
                print(f"  No se ejecuta: se necesitan {min_new} NUEVAS desde la ultima recalibracion (tienes {count_new}).")
                return

        # Dataset: todas o últimos N días
        trade_features = await db.get_training_dataset(days=training_days)
        print(f"  Dataset de entrenamiento: {len(trade_features)} trade_features" + (f" (ultimos {training_days} dias)" if training_days else " (todas)"))

        if len(trade_features) < min_new:
            print(f"  No se ejecuta: dataset tiene {len(trade_features)} < {min_new}.")
            return

        hazard_model = HazardModel()
        pump_model = PumpModel()
        hazard_model.set_database(db)
        pump_model.set_database(db)
        await hazard_model.load_params()
        await pump_model.load_params()

        # Hazard
        hazard_trainer = hazard_model.get_trainer()
        hazard_result = hazard_trainer.train(trade_features)
        if hazard_result.success:
            hazard_model.update_from_trainer()
            await hazard_model.save_params()
            print(f"  Hazard: OK, loss improved {hazard_result.improvement:.1%}, n={hazard_result.n_samples}")
        else:
            print(f"  Hazard: {hazard_result.message}")

        # Pump
        pump_trainer = pump_model.get_trainer()
        pump_result = pump_trainer.train(trade_features)
        if pump_result.success:
            pump_model.update_from_trainer()
            await pump_model.save_params()
            print(f"  Pump: OK, loss improved {pump_result.improvement:.1%}, n={pump_result.n_samples}")
        else:
            print(f"  Pump: {pump_result.message}")

        # G buckets
        g_update = await pump_model.update_g_buckets(trade_features)
        print(f"  G buckets: {g_update.get('updated', [])}")

        await db.set_last_recalibration_at()
        print("\n  Recalibracion completada. Estado guardado (proxima vez: 30 nuevas desde ahora).")
    except Exception as e:
        logger.exception(e)
        print(f"\n  ERROR: {e}")
    finally:
        await db.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Recalibracion v2.0 standalone (30 trade_features NUEVAS desde ultima vez)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ejecutar aunque no haya 30 nuevas (requiere al menos 30 totales)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Opcional: entrenar solo con trade_features de los ultimos N dias (default: todas)",
    )
    args = parser.parse_args()
    asyncio.run(run_recalibration(force=args.force, training_days=args.days))


if __name__ == "__main__":
    main()
