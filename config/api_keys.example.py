"""
Claves API para servicios externos.
Copia este archivo como api_keys.py y añade tus claves.
"""

import os
from dotenv import load_dotenv

load_dotenv()

API_KEYS = {
    "solscan": os.getenv("SOLSCAN_API_KEY", "tu_api_key_aqui"),
    "helius": os.getenv("HELIUS_API_KEY", "tu_api_key_aqui"),
}
