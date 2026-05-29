"""
KyberExceptions.py
──────────────────
Typed exception hierarchy for the Kyber KEM pipeline.
Every public method in KyberCoreEngine raises one of these instead of
letting raw Python errors bubble up.

V7 additions
────────────
  • FOTransformError       — re-encryption mismatch (c != c') in decapsulation
  • ImplicitRejectionError — constant-time branch of FO; carries a garbage_key
                             so callers never see a timing difference
  • SoftDecodeError        — soft-decision decoder failed even after majority vote
  • WeakBitWarning         — non-fatal; bits resolved by majority vote
  • DynamicDvError         — dynamic dv selector found no safe compression level
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


# ── FO Transform errors ──────────────────────────────────────────────────────

class FOTransformError(DecryptionError):
    """
    Raised internally when the FO re-encryption check fails (c != c').

    This is NOT a programming error — it signals that the ciphertext was
    tampered with or corrupted.  KyberCoreEngine.decrypt() catches this and
    raises ImplicitRejectionError instead of leaking oracle information.

    Attributes
    ----------
    u_mismatch    : bool  — u component differed from re-encrypted u'
    v_mismatch    : bool  — v component differed from re-encrypted v'
    bad_positions : list  — coefficient indices of first detected mismatch
    """
    def __init__(
        self,
        u_mismatch: bool = False,
        v_mismatch: bool = False,
        bad_positions=None,
    ):
        self.u_mismatch    = u_mismatch
        self.v_mismatch    = v_mismatch
        self.bad_positions = bad_positions or []
        detail = []
        if u_mismatch:
            detail.append("u component mismatch")
        if v_mismatch:
            detail.append("v component mismatch")
        super().__init__(
            f"FO re-encryption check failed: {', '.join(detail) or 'unknown'}. "
            "Ciphertext integrity violated."
        )


class ImplicitRejectionError(DecryptionError):
    """
    Raised when FO implicit rejection is triggered (c != c').

    The caller MUST NOT retry with the same ciphertext.  The attribute
    `garbage_key` is a deterministic pseudo-random 32-byte value derived from
    the rejection seed z and the serialised ciphertext.  Return it to the
    caller in place of the real shared secret — the adversary observes no
    timing or error-code difference from a successful decapsulation.

    Attributes
    ----------
    garbage_key : bytes — H(z || serialised_ciphertext), exactly 32 bytes
    """
    def __init__(self, garbage_key: bytes):
        self.garbage_key = garbage_key
        super().__init__(
            "Implicit rejection triggered: ciphertext failed FO integrity check. "
            "Use .garbage_key as the opaque shared secret to avoid oracle leakage."
        )


# ── Soft-decision decoding errors ────────────────────────────────────────────

class SoftDecodeError(DecryptionError):
    """
    Raised when the soft-decision decoder cannot resolve all bits even after
    majority voting.

    Attributes
    ----------
    weak_bit_count : int   — number of bits below threshold after all passes
    threshold      : float — confidence threshold (0.0 – 1.0)
    attempt        : int   — which retry pass triggered the raise
    """
    def __init__(self, weak_bit_count: int, threshold: float, attempt: int = 1):
        self.weak_bit_count = weak_bit_count
        self.threshold      = threshold
        self.attempt        = attempt
        super().__init__(
            f"Soft-decision decode failed: {weak_bit_count} bit(s) remained below "
            f"confidence threshold {threshold:.3f} after attempt {attempt}. "
            "Possible excessive noise or corrupted ciphertext."
        )


class WeakBitWarning(UserWarning):
    """
    Non-fatal warning emitted when majority voting resolved low-confidence bits.
    Not raised — attach to decode result metadata.

    Attributes
    ----------
    positions   : list[int]   — bit positions resolved by majority vote
    confidences : list[float] — per-bit confidence at those positions (0–1)
    """
    def __init__(self, positions, confidences):
        self.positions   = list(positions)
        self.confidences = list(confidences)
        super().__init__(
            f"{len(positions)} bit(s) resolved by majority vote "
            f"(min confidence {min(confidences, default=0.0):.3f}). "
            "Consider increasing dv or reducing channel noise."
        )


# ── Dynamic dv errors ────────────────────────────────────────────────────────

class DynamicDvError(KyberError):
    """
    Raised by the dynamic dv selector when no compression level satisfies both
    the noise budget AND the bandwidth constraint simultaneously.

    Attributes
    ----------
    observed_max_noise : int — maximum absolute centred noise observed
    noise_budget       : int — Q // 4 (hard LWE error-correction ceiling)
    min_dv_tried       : int — smallest dv evaluated
    max_dv_tried       : int — largest dv evaluated
    """
    def __init__(
        self,
        observed_max_noise: int,
        noise_budget: int,
        min_dv_tried: int,
        max_dv_tried: int,
    ):
        self.observed_max_noise = observed_max_noise
        self.noise_budget       = noise_budget
        self.min_dv_tried       = min_dv_tried
        self.max_dv_tried       = max_dv_tried
        super().__init__(
            f"Dynamic dv selection failed: observed max noise {observed_max_noise} "
            f"exceeds budget {noise_budget} for all dv in "
            f"[{min_dv_tried}, {max_dv_tried}]. "
            "Increase security parameters or reduce channel noise."
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