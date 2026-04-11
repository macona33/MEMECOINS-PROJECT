"""Hot wallet desde PRIVATE_KEY (nunca loguear la clave)."""

import json
import os
from typing import Optional

import base58
from dotenv import load_dotenv
from loguru import logger
from solders.keypair import Keypair

load_dotenv()


def load_keypair_from_env(env_var: str = "PRIVATE_KEY") -> Optional[Keypair]:
    """
    Carga Keypair desde env.
    - Preferido: base58 de 64 bytes (secret key completa).
    - Fallback: JSON array de 64 enteros.
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        logger.warning(f"{env_var} no está definida")
        return None

    try:
        if raw.startswith("["):
            arr = json.loads(raw)
            if not isinstance(arr, list) or len(arr) not in (32, 64):
                raise ValueError("JSON key: se esperan 32 o 64 bytes")
            secret = bytes(int(x) & 0xFF for x in arr)
            return Keypair.from_bytes(secret)

        decoded = base58.b58decode(raw)
        if len(decoded) not in (32, 64):
            raise ValueError(f"Longitud de clave inválida: {len(decoded)}")
        return Keypair.from_bytes(decoded)
    except Exception as e:
        logger.error(f"No se pudo decodificar {env_var}: {e}")
        raise ValueError("PRIVATE_KEY inválida") from e
