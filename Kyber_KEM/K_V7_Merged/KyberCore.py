"""
KyberCore.py
────────────
Central Hardware FSM — combines:
  • NTT-accelerated arithmetic  (O(N log N), A stored in NTT domain, A-cache)
  • Full typed error handling via KyberExceptions
  • Integrated CiphertextCompressor (optional, off by default)
  • Ciphertext serialisation via SerializationUnit (du/dv compression)
  • prepare_payload() helper for arbitrary Python values → bytes chunks
  • encrypt_chunked() / decrypt_chunked() for messages longer than chunk_size

Kyber variant defaults:
    Kyber-512  : k=2, du=10, dv=4  →  768 byte ciphertext
    Kyber-768  : k=3, du=10, dv=4  → 1088 byte ciphertext
    Kyber-1024 : k=4, du=11, dv=5  → 1568 byte ciphertext
"""

import hashlib
import os
import struct
from typing import Any, List, Optional, Tuple

from .ArithmeticUnit import ArithmeticUnit, ntt, intt, ntt_mul
from .CiphertextCompressor import CiphertextCompressor
from .KyberExceptions import (
    CiphertextListError,
    DecryptionError,
    DecompressionError,
    EmptyMessageError,
    EncryptionError,
    InvalidCiphertextError,
    InvalidMessageTypeError,
    InvalidPublicKeyError,
    InvalidSecretKeyError,
    InvalidSeedError,
    KeyGenError,
    MessageSizeError,
    ParameterError,
)
from .SamplingUnit import SamplingUnit
from .SerializationUnit import SerializationUnit
from .VectorMatrixMultiplier import VectorMatrixMultiplier

Poly = List[int]
Vec  = List[Poly]


# ── parameter validation ─────────────────────────────────────────────────────

def _validate_params(n: int, q: int, eta: int, k: int) -> None:
    if not isinstance(n, int) or n < 2 or (n & (n - 1)) != 0:
        raise ParameterError("n", n, "Must be a power of 2 ≥ 2.")
    if not isinstance(q, int) or q < 3:
        raise ParameterError("q", q, "Must be an integer ≥ 3.")
    if not isinstance(eta, int) or eta < 1:
        raise ParameterError("eta", eta, "Must be a positive integer.")
    if not isinstance(k, int) or k < 1:
        raise ParameterError("k_dimension", k, "Must be a positive integer.")


# ── payload helpers ──────────────────────────────────────────────────────────

def _to_bytes(value: Any) -> bytes:
    """
    Coerce common Python types to bytes.

    Wire format adds a 1-byte type tag + 4-byte big-endian length prefix so
    that decrypt_chunked() can recover the exact original value regardless of
    trailing-zero bytes or leading-zero bits.

    Tag values:
        0x01  raw bytes / bytearray
        0x02  str (UTF-8)
        0x03  int (big-endian, min 1 byte; sign stored as leading byte)
        0x04  float (IEEE 754 double, 8 bytes)
        0x05  list / tuple (recursively encoded items, length-prefixed)
    """
    if isinstance(value, (bytes, bytearray)):
        if not value: return b""
        raw = bytes(value)
        return b"\x01" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, str):
        if not value: return b""
        raw = value.encode("utf-8")
        return b"\x02" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, int):
        sign = 0x01 if value < 0 else 0x00
        mag  = abs(value)
        byte_len = max(1, (mag.bit_length() + 7) // 8)
        raw  = bytes([sign]) + mag.to_bytes(byte_len, "big")
        return b"\x03" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, float):
        raw = struct.pack(">d", value)
        return b"\x04" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, (list, tuple)):
        if not value: return b""
        inner = b"".join(_to_bytes(item) for item in value)
        return b"\x05" + struct.pack(">I", len(inner)) + inner
    raise InvalidMessageTypeError(type(value))


def _from_bytes(data: bytes, output_type: str):
    """
    Decode a tagged byte string produced by _to_bytes() back to its original
    Python value.  Falls back to output_type hint for untagged raw chunks.
    """
    if not data:
        if output_type == "bytes":  return b""
        if output_type == "str":    return ""
        if output_type == "int":    return 0
        if output_type == "float":  return 0.0
        return data

    tag = data[0]
    if tag in (0x01, 0x02, 0x03, 0x04, 0x05) and len(data) >= 5:
        declared_len = struct.unpack_from(">I", data, 1)[0]
        if declared_len <= len(data) - 5:
            payload = data[5 : 5 + declared_len]
            if tag == 0x01:
                return payload
            if tag == 0x02:
                return payload.decode("utf-8")
            if tag == 0x03:
                if len(payload) == 0:
                    return 0
                sign = payload[0]
                mag  = int.from_bytes(payload[1:], "big") if len(payload) > 1 else 0
                return -mag if sign else mag
            if tag == 0x04:
                if len(payload) < 8:
                    payload = payload.ljust(8, b"\x00")
                return struct.unpack(">d", payload[:8])[0]
            if tag == 0x05:
                return _from_bytes(payload, output_type)

    # Legacy / raw path
    raw = data.rstrip(b"\x00")
    if output_type == "bytes":
        return raw
    if output_type == "str":
        return raw.decode("utf-8")
    if output_type == "int":
        return int.from_bytes(raw, "big") if raw else 0
    if output_type == "float":
        if len(raw) < 8:
            raw = raw.ljust(8, b"\x00")
        return struct.unpack(">d", raw[:8])[0]
    return raw


def _chunk_bytes(data: bytes, chunk_size: int) -> List[bytes]:
    """Split bytes into fixed-size blocks, padding the last one with NUL."""
    if not data:
        raise EmptyMessageError()
    chunks = []
    for offset in range(0, len(data), chunk_size):
        block = data[offset : offset + chunk_size]
        if len(block) < chunk_size:
            block = block.ljust(chunk_size, b"\x00")
        chunks.append(block)
    return chunks


# ── core engine ──────────────────────────────────────────────────────────────

class KyberCoreEngine:
    """
    Central Hardware FSM scheduling all Kyber operations.

    NTT Optimisations (from V5)
    ───────────────────────────
    1. NTT multiplication    — O(N log N) vs O(N²) schoolbook
    2. A stored in NTT domain — generate_A_matrix emits NTT polys directly
    3. A_cache keyed on rho  — identical public key reuses the same A matrix
    4. r and s NTT'd once    — reused across all dot-products in encrypt/decrypt

    Error Handling (from V6)
    ────────────────────────
    Every public method raises a typed KyberException on invalid input.

    Parameters
    ----------
    n, q, eta     : ring parameters
    k_dimension   : module rank (2 → Kyber-512, 3 → 768, 4 → 1024)
    compress      : enable CiphertextCompressor (zlib+quantisation, default False)
    d_u, d_v      : Kyber spec compression bits for u/v polynomials
    use_zlib      : zlib on top of quantisation (only when compress=True)
    """

    def __init__(
        self,
        n: int,
        q: int,
        eta: int,
        k_dimension: int,
        compress: bool = False,
        d_u: int = 10,
        d_v: int = 4,
        use_zlib: bool = True,
    ):
        _validate_params(n, q, eta, k_dimension)

        self.k          = k_dimension
        self.alu        = ArithmeticUnit(n, q)
        self.sampler    = SamplingUnit(n, q, eta)
        self.vmm        = VectorMatrixMultiplier(self.alu)
        self.serializer = SerializationUnit(n, q)
        self._du        = d_u
        self._dv        = d_v
        self._compress  = compress
        self._compressor = CiphertextCompressor(q=q, d_u=d_u, d_v=d_v, use_zlib=use_zlib)
        self.A_cache: dict = {}   # rho → NTT-domain A matrix

    # ── A-matrix cache (V5 optimisation) ─────────────────────────────────────

    def _get_A(self, rho: bytes) -> list:
        """Return cached NTT-domain A matrix; generate only once per rho."""
        if rho not in self.A_cache:
            self.A_cache[rho] = self.vmm.generate_A_matrix(
                self.sampler, rho, self.k, store_ntt=True)
        return self.A_cache[rho]

    # ── key generation ────────────────────────────────────────────────────────

    def keygen(self, seed: Optional[bytes] = None) -> Tuple[tuple, Vec]:
        """
        Generate a (public_key, secret_key) pair.
        seed must be exactly 32 bytes (or None for a random seed).
        """
        try:
            if seed is None:
                seed = os.urandom(32)
            if not isinstance(seed, (bytes, bytearray)):
                raise InvalidSeedError(f"Expected bytes, got {type(seed).__name__}.")
            if len(seed) != 32:
                raise InvalidSeedError(f"Expected 32 bytes, got {len(seed)}.")

            h     = hashlib.sha3_512(seed).digest()
            rho   = h[:32]
            sigma = h[32:]

            A_ntt = self._get_A(rho)

            s     = [self.sampler.sample_cbd(sigma, idx)          for idx in range(self.k)]
            e     = [self.sampler.sample_cbd(sigma, self.k + idx) for idx in range(self.k)]
            s_ntt = [ntt(si) for si in s]

            # t = A·s + e  (A and s both in NTT domain — no redundant fwd NTT)
            As = self.vmm.mat_vec_mul(A_ntt, s_ntt, A_in_ntt=True, v_in_ntt=True)
            t  = self.vmm.vec_add(As, e)

            # s returned in coefficient domain for decryption
            return (rho, t), s

        except (InvalidSeedError, ParameterError):
            raise
        except Exception as exc:
            raise KeyGenError(f"Key generation failed: {exc}") from exc

    # ── single-chunk encrypt / decrypt (low-level) ────────────────────────────

    def encrypt(
        self,
        public_key: tuple,
        message: bytes,
        r_seed: Optional[bytes] = None,
    ):
        """
        Encrypt a single message chunk (must be exactly chunk_size bytes).
        Returns compressed bytes if compress=True, else a raw (u, v) tuple.
        """
        try:
            # --- validate public key ---
            if not (isinstance(public_key, tuple) and len(public_key) == 2):
                raise InvalidPublicKeyError("Must be a (rho, t) tuple.")
            rho, t = public_key
            if not isinstance(rho, (bytes, bytearray)) or len(rho) != 32:
                raise InvalidPublicKeyError("rho must be 32 bytes.")
            if not isinstance(t, list) or len(t) != self.k:
                raise InvalidPublicKeyError(
                    f"t must be a list of {self.k} polynomials; "
                    f"got length {len(t) if isinstance(t, list) else 'N/A'}."
                )

            # --- validate message ---
            if not isinstance(message, (bytes, bytearray)):
                raise InvalidMessageTypeError(type(message))
            if len(message) == 0:
                raise EmptyMessageError()
            if len(message) != self.serializer.chunk_size:
                raise MessageSizeError(len(message), self.serializer.chunk_size)

            if r_seed is None:
                r_seed = os.urandom(32)

            A_ntt  = self._get_A(rho)
            AT_ntt = self.vmm.mat_transpose(A_ntt)

            r  = [self.sampler.sample_cbd(r_seed, idx)          for idx in range(self.k)]
            e1 = [self.sampler.sample_cbd(r_seed, self.k + idx) for idx in range(self.k)]
            e2 = self.sampler.sample_cbd(r_seed, 2 * self.k)

            # NTT r and t once, reuse for both AT·r and t·r
            r_ntt = [ntt(ri) for ri in r]
            t_ntt = [ntt(ti) for ti in t]

            u  = self.vmm.vec_add(
                     self.vmm.mat_vec_mul(AT_ntt, r_ntt, A_in_ntt=True, v_in_ntt=True),
                     e1)
            tr = self.vmm.vec_dot(t_ntt, r_ntt, a_in_ntt=True, b_in_ntt=True)
            v  = self.alu.poly_add(
                     self.alu.poly_add(tr, e2),
                     self.serializer.encode_message_to_poly(message))

            ct = (u, v)
            if self._compress:
                return self._compressor.compress(ct)
            return ct

        except (
            InvalidPublicKeyError, InvalidMessageTypeError,
            EmptyMessageError, MessageSizeError,
        ):
            raise
        except Exception as exc:
            raise EncryptionError(f"Encryption failed: {exc}") from exc

    def decrypt(
        self,
        secret_key: Vec,
        ciphertext,
    ) -> bytes:
        """
        Decrypt a single ciphertext (raw tuple or compressed bytes).
        Returns exactly chunk_size raw bytes.
        """
        try:
            # --- validate secret key ---
            if not isinstance(secret_key, list) or len(secret_key) != self.k:
                raise InvalidSecretKeyError(
                    f"Expected list of {self.k} polynomials."
                )

            # --- decompress if needed ---
            if isinstance(ciphertext, (bytes, bytearray)):
                try:
                    ciphertext = self._compressor.decompress(ciphertext)
                except DecompressionError:
                    raise
                except Exception as exc:
                    raise InvalidCiphertextError(
                        f"Cannot decompress ciphertext: {exc}"
                    ) from exc

            if not (isinstance(ciphertext, tuple) and len(ciphertext) == 2):
                raise InvalidCiphertextError("Must be a (u, v) tuple.")

            u, v = ciphertext
            if not isinstance(u, list) or len(u) != self.k:
                raise InvalidCiphertextError(
                    f"u must be a list of {self.k} polynomials."
                )
            if not isinstance(v, list) or len(v) != self.serializer.N:
                raise InvalidCiphertextError(
                    f"v must be a polynomial of length {self.serializer.N}."
                )

            # NTT s and u once, reuse for the dot product
            s_ntt = [ntt(si) for si in secret_key]
            u_ntt = [ntt(ui) for ui in u]
            su    = self.vmm.vec_dot(s_ntt, u_ntt, a_in_ntt=True, b_in_ntt=True)
            w     = self.alu.poly_sub(v, su)
            return self.serializer.decode_poly_to_message(w)

        except (
            InvalidSecretKeyError, InvalidCiphertextError,
            DecompressionError,
        ):
            raise
        except Exception as exc:
            raise DecryptionError(f"Decryption failed: {exc}") from exc

    # ── high-level chunked API ────────────────────────────────────────────────

    def prepare_payload(self, value: Any) -> List[bytes]:
        """
        Convert any supported Python value (str, int, float, bytes, list …)
        to a list of chunk_size byte blocks ready for encrypt_chunked().
        """
        raw = _to_bytes(value)
        return _chunk_bytes(raw, self.serializer.chunk_size)

    def encrypt_chunked(
        self,
        public_key: tuple,
        value: Any,
        r_seed: Optional[bytes] = None,
    ) -> List:
        """
        Encrypt an arbitrary value (any type supported by prepare_payload).
        Returns a list of ciphertexts — one per chunk.
        """
        chunks = self.prepare_payload(value)
        result = []
        for i, chunk in enumerate(chunks):
            chunk_seed = (
                hashlib.sha256((r_seed or b"") + i.to_bytes(4, "big")).digest()
                if r_seed is not None
                else None
            )
            result.append(self.encrypt(public_key, chunk, chunk_seed))
        return result

    def decrypt_chunked(
        self,
        secret_key: Vec,
        ciphertexts: List,
        output_type: str = "bytes",
    ):
        """
        Decrypt a list of ciphertexts produced by encrypt_chunked().

        output_type is used as a fallback hint when the payload has no type tag.
        Supported: "bytes" | "str" | "int" | "float"
        """
        if not isinstance(ciphertexts, list):
            raise CiphertextListError(type(ciphertexts))

        raw = b"".join(self.decrypt(secret_key, ct) for ct in ciphertexts)
        return _from_bytes(raw, output_type)
