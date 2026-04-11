"""Firma de transacciones Versioned (Jupiter v6)."""

import base64

from loguru import logger
from solders import message as sm
from solders.keypair import Keypair
from solders.signature import Signature
from solders.transaction import VersionedTransaction


def sign_jupiter_versioned_transaction(
    swap_tx_b64: str,
    keypair: Keypair,
) -> VersionedTransaction:
    raw = base64.b64decode(swap_tx_b64)
    vtx = VersionedTransaction.from_bytes(raw)
    msg = vtx.message
    keys = list(msg.account_keys)
    pk = keypair.pubkey()
    n_req = int(msg.header.num_required_signatures)
    signer_keys = keys[:n_req]

    user_sig = keypair.sign_message(sm.to_bytes_versioned(msg))

    # Caso habitual Jupiter: un solo firmante (= fee payer).
    if n_req == 1:
        if signer_keys and signer_keys[0] != pk:
            logger.warning(
                "El fee payer de la tx (%s) no coincide con tu wallet (%s). "
                "Revisa userPublicKey en la petición swap.",
                signer_keys[0] if signer_keys else None,
                pk,
            )
        return VersionedTransaction.populate(msg, [user_sig])

    if pk not in signer_keys:
        if pk in keys:
            logger.warning(
                "Tu pubkey está en la tx pero no entre los %s firmantes exigidos; "
                "usando VersionedTransaction(msg, [keypair])",
                n_req,
            )
        else:
            logger.warning(
                "Pubkey del wallet no está entre los firmantes; firmando con VersionedTransaction(msg, [keypair])"
            )
        return VersionedTransaction(msg, [keypair])

    idx = signer_keys.index(pk)
    sigs = list(vtx.signatures)
    if len(sigs) < n_req:
        sigs = sigs + [Signature.default()] * (n_req - len(sigs))
    if idx >= len(sigs):
        sigs = sigs + [Signature.default()] * (idx + 1 - len(sigs))
    sigs[idx] = user_sig
    return VersionedTransaction.populate(msg, sigs[:n_req])
