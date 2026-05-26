"""
KyberExceptions.py
──────────────────
Typed exception hierarchy for the Kyber KEM pipeline.
Every public method in KyberCoreEngine raises one of these instead of
letting raw Python errors bubble up.
"""


class KyberError(Exception):
    """Base class for all Kyber errors."""


# ── Key-generation errors ────────────────────────────────────────────────────

class KeyGenError(KyberError):
    """Raised when the key-generation pipeline fails."""


class InvalidSeedError(KeyGenError):
    """Seed is None, wrong type, or wrong length."""
    def __init__(self, detail: str = ""):
        super().__init__(f"Invalid key-generation seed. {detail}".strip())


# ── Encryption errors ────────────────────────────────────────────────────────

class EncryptionError(KyberError):
    """Raised when the encryption pipeline fails."""


class InvalidPublicKeyError(EncryptionError):
    """Public key has wrong structure or dimensions."""
    def __init__(self, detail: str = ""):
        super().__init__(f"Invalid public key. {detail}".strip())


class MessageSizeError(EncryptionError):
    """Message chunk is not exactly chunk_size bytes."""
    def __init__(self, got: int, expected: int):
        super().__init__(
            f"Message chunk must be exactly {expected} bytes; got {got}."
        )


class InvalidMessageTypeError(EncryptionError):
    """Input could not be coerced to bytes."""
    def __init__(self, typ: type):
        super().__init__(
            f"Cannot encrypt value of type '{typ.__name__}'. "
            "Convert to bytes first or use KyberCoreEngine.prepare_payload()."
        )


class EmptyMessageError(EncryptionError):
    """Empty payload supplied."""
    def __init__(self):
        super().__init__("Cannot encrypt an empty message.")


# ── Decryption errors ────────────────────────────────────────────────────────

class DecryptionError(KyberError):
    """Raised when the decryption pipeline fails."""


class InvalidSecretKeyError(DecryptionError):
    """Secret key has wrong structure or dimensions."""
    def __init__(self, detail: str = ""):
        super().__init__(f"Invalid secret key. {detail}".strip())


class InvalidCiphertextError(DecryptionError):
    """Ciphertext is malformed, truncated, or tampered with."""
    def __init__(self, detail: str = ""):
        super().__init__(f"Invalid ciphertext. {detail}".strip())


class CiphertextListError(DecryptionError):
    """Expected a list of ciphertexts (chunked mode) but got something else."""
    def __init__(self, typ: type):
        super().__init__(
            f"Expected a list of ciphertexts; got '{typ.__name__}'."
        )


# ── Compression errors ───────────────────────────────────────────────────────

class CompressionError(KyberError):
    """Raised by the CiphertextCompressor when packing/unpacking fails."""


class DecompressionError(KyberError):
    """Raised when a compressed blob cannot be decompressed."""
    def __init__(self, detail: str = ""):
        super().__init__(f"Decompression failed. {detail}".strip())


# ── Parameter errors ─────────────────────────────────────────────────────────

class ParameterError(KyberError):
    """Raised for invalid N / Q / eta / k_dimension combinations."""
    def __init__(self, param: str, value, detail: str = ""):
        super().__init__(
            f"Invalid parameter '{param}={value}'. {detail}".strip()
        )