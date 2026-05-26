"""
KyberCore.py
────────────
Central Hardware FSM — combines:
  • NTT-accelerated arithmetic  (O(N log N), A stored in NTT domain, A-cache)
  • Full typed error handling via KyberExceptions
  • Integrated CiphertextCompressor (optional, off by default)
  • Ciphertext serialisation via SerializationUnit (du/dv compression)
  • prepare_payload() helper for arbitrary Python values -> bytes chunks
  • encrypt_chunked() / decrypt_chunked() for messages longer than chunk_size

V7 additions
────────────
  • FO Transform verification in decrypt():
      1. Decrypt ciphertext  ->  tentative message p'
      2. Re-derive randomness  L' = H+(p')
      3. Re-encrypt p' with L'  ->  (u', v')
      4. Constant-time compare  (u, v) == (u', v')
         OK    -> return real shared secret
         FAIL  -> raise ImplicitRejectionError(garbage_key=H(z || ct_bytes))
  • Soft-decision decoding: decrypt() and decrypt_soft() both available
  • Dynamic dv selection via SerializationUnit.select_dv_dynamic()
  • Noise accumulator for adaptive dv tuning across calls

Kyber variant defaults:
    Kyber-512  : k=2, du=10, dv=4  ->  768 byte ciphertext
    Kyber-768  : k=3, du=10, dv=4  -> 1088 byte ciphertext
    Kyber-1024 : k=4, du=11, dv=5  -> 1568 byte ciphertext
"""

import hashlib
import hmac
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
    FOTransformError,
    ImplicitRejectionError,
    InvalidCiphertextError,
    InvalidMessageTypeError,
    InvalidPublicKeyError,
    InvalidSecretKeyError,
    InvalidSeedError,
    KeyGenError,
    MessageSizeError,
    ParameterError,
    SoftDecodeError,
)
from .SamplingUnit import SamplingUnit
from .SerializationUnit import SerializationUnit, SoftDecodeResult, select_dv_dynamic
from .VectorMatrixMultiplier import VectorMatrixMultiplier

Poly = List[int]
Vec  = List[Poly]


# ── parameter validation ─────────────────────────────────────────────────────

def _validate_params(n: int, q: int, eta: int, k: int) -> None:
    if not isinstance(n, int) or n < 2 or (n & (n - 1)) != 0:
        raise ParameterError("n", n, "Must be a power of 2 >= 2.")
    if not isinstance(q, int) or q < 3:
        raise ParameterError("q", q, "Must be an integer >= 3.")
    if not isinstance(eta, int) or eta < 1:
        raise ParameterError("eta", eta, "Must be a positive integer.")
    if not isinstance(k, int) or k < 1:
        raise ParameterError("k_dimension", k, "Must be a positive integer.")


# ── payload helpers ──────────────────────────────────────────────────────────

def _to_bytes(value: Any) -> bytes:
    """
    Coerce common Python types to bytes.

    Wire format: 1-byte type tag + 4-byte big-endian length + payload.
    Tags: 0x01=bytes 0x02=str 0x03=int 0x04=float 0x05=list/tuple
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
    """Decode a tagged byte string produced by _to_bytes() back to its Python value."""
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
            if tag == 0x01: return payload
            if tag == 0x02: return payload.decode("utf-8")
            if tag == 0x03:
                if not payload: return 0
                sign = payload[0]
                mag  = int.from_bytes(payload[1:], "big") if len(payload) > 1 else 0
                return -mag if sign else mag
            if tag == 0x04:
                if len(payload) < 8: payload = payload.ljust(8, b"\x00")
                return struct.unpack(">d", payload[:8])[0]
            if tag == 0x05:
                return _from_bytes(payload, output_type)

    raw = data.rstrip(b"\x00")
    if output_type == "bytes":  return raw
    if output_type == "str":    return raw.decode("utf-8")
    if output_type == "int":    return int.from_bytes(raw, "big") if raw else 0
    if output_type == "float":
        if len(raw) < 8: raw = raw.ljust(8, b"\x00")
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


# ── FO transform helpers ─────────────────────────────────────────────────────

def _ct_to_bytes(u: Vec, v: Poly, serializer: SerializationUnit,
                 du: int, dv: int) -> bytes:
    """Serialise a (u, v) ciphertext pair to a canonical byte string."""
    return serializer.compress_ciphertext(u, v, du=du, dv=dv)


def _constant_time_compare_poly(a: Poly, b: Poly) -> bool:
    """
    Constant-time polynomial equality.  Uses hmac.compare_digest on packed
    bytes so the runtime does not depend on the position of the first mismatch.
    """
    if len(a) != len(b):
        return False
    ba = struct.pack(f"<{len(a)}H", *(x % 3329 for x in a))
    bb = struct.pack(f"<{len(b)}H", *(x % 3329 for x in b))
    return hmac.compare_digest(ba, bb)


def _constant_time_compare_vec(u: Vec, u2: Vec) -> bool:
    """Constant-time vector equality."""
    if len(u) != len(u2):
        return False
    result = True
    for a, b in zip(u, u2):
        result &= _constant_time_compare_poly(a, b)
    return result


# ── core engine ──────────────────────────────────────────────────────────────

class KyberCoreEngine:
    """
    Central Hardware FSM scheduling all Kyber operations.

    NTT Optimisations
    ─────────────────
    1. NTT multiplication    — O(N log N) vs O(N^2) schoolbook
    2. A stored in NTT domain — generate_A_matrix emits NTT polys directly
    3. A_cache keyed on rho  — identical public key reuses the same A matrix
    4. r and s NTT'd once    — reused across all dot-products in encrypt/decrypt

    FO Transform (V7)
    ─────────────────
    decrypt() now performs the full Fujisaki-Okamoto re-encryption check:

        p'      <- hard-decode(v - s·u)
        K', L'  <- H+(p')
        u', v'  <- Encrypt(pk, p'; L')
        if (u, v) == (u', v'):
            return K'  [or p' in this IND-CPA layer]
        else:
            raise ImplicitRejectionError(garbage_key = H(z || ct_bytes))

    The check uses constant-time comparison (hmac.compare_digest) so no
    timing side-channel leaks the mismatch position.

    Soft Decoding (V7)
    ──────────────────
    decrypt_soft() returns a SoftDecodeResult with per-bit confidence scores
    and weak-bit positions resolved by majority voting.

    Dynamic dv (V7)
    ───────────────
    Call select_dv() to pick the smallest safe compression level based on
    accumulated noise observations, security level, and FPGA bandwidth profile.

    Parameters
    ----------
    n, q, eta       : ring parameters
    k_dimension     : module rank (2->512, 3->768, 4->1024)
    compress        : enable CiphertextCompressor (zlib+quantisation)
    d_u, d_v        : compression bits for u/v polynomials
    use_zlib        : zlib on top of quantisation (only when compress=True)
    rejection_seed  : 32-byte z for implicit rejection; random if None
    fo_enabled      : enable FO transform check in decrypt() (default True)
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
        rejection_seed: Optional[bytes] = None,
        fo_enabled: bool = True,
    ):
        _validate_params(n, q, eta, k_dimension)

        self.k           = k_dimension
        self.alu         = ArithmeticUnit(n, q)
        self.sampler     = SamplingUnit(n, q, eta)
        self.vmm         = VectorMatrixMultiplier(self.alu)
        self.serializer  = SerializationUnit(n, q)
        self._du         = d_u
        self._dv         = d_v
        self._compress   = compress
        self._compressor = CiphertextCompressor(q=q, d_u=d_u, d_v=d_v, use_zlib=use_zlib)
        self.A_cache: dict = {}

        # FO transform
        self._fo_enabled = fo_enabled
        if rejection_seed is None:
            rejection_seed = os.urandom(32)
        if not isinstance(rejection_seed, (bytes, bytearray)) or len(rejection_seed) != 32:
            raise ParameterError(
                "rejection_seed", rejection_seed,
                "Must be exactly 32 bytes."
            )
        self._z = bytes(rejection_seed)

        # Noise accumulator for dynamic dv selection
        self._noise_samples: List[int] = []
        self._noise_window  = 512   # keep last N centred-noise observations

    # ── A-matrix cache ────────────────────────────────────────────────────────

    def _get_A(self, rho: bytes):
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

            As = self.vmm.mat_vec_mul(A_ntt, s_ntt, A_in_ntt=True, v_in_ntt=True)
            t  = self.vmm.vec_add(As, e)

            return (rho, t), s

        except (InvalidSeedError, ParameterError):
            raise
        except Exception as exc:
            raise KeyGenError(f"Key generation failed: {exc}") from exc

    # ── internal encrypt (low-level, returns raw (u,v) always) ───────────────

    def _encrypt_raw(
        self,
        public_key: tuple,
        message: bytes,
        r_seed: bytes,
    ) -> Tuple[Vec, Poly]:
        """
        Core encryption returning (u, v) always (no compression).
        Used internally for FO re-encryption and by encrypt().
        """
        rho, t = public_key
        A_ntt  = self._get_A(rho)
        AT_ntt = self.vmm.mat_transpose(A_ntt)

        r  = [self.sampler.sample_cbd(r_seed, idx)          for idx in range(self.k)]
        e1 = [self.sampler.sample_cbd(r_seed, self.k + idx) for idx in range(self.k)]
        e2 = self.sampler.sample_cbd(r_seed, 2 * self.k)

        r_ntt = [ntt(ri) for ri in r]
        t_ntt = [ntt(ti) for ti in t]

        u  = self.vmm.vec_add(
                 self.vmm.mat_vec_mul(AT_ntt, r_ntt, A_in_ntt=True, v_in_ntt=True),
                 e1)
        tr = self.vmm.vec_dot(t_ntt, r_ntt, a_in_ntt=True, b_in_ntt=True)
        v  = self.alu.poly_add(
                 self.alu.poly_add(tr, e2),
                 self.serializer.encode_message_to_poly(message))
        return u, v

    # ── FO re-encryption check ────────────────────────────────────────────────

    def _fo_check(
        self,
        public_key: tuple,
        p_prime: bytes,
        u: Vec,
        v: Poly,
        ct_bytes: bytes,
    ) -> None:
        """
        Perform the FO transform re-encryption check.

        Re-derives randomness L' from H+(p'), re-encrypts p', and compares
        (u', v') with (u, v) using constant-time comparison.

        Raises
        ------
        ImplicitRejectionError  — if (u, v) != (u', v')
            .garbage_key = SHA3-256(z || ct_bytes)
        """
        # Derive (K', L') from the tentative plaintext
        h_plus  = hashlib.sha3_512(p_prime).digest()
        # K_prime = h_plus[:32]  # shared secret — unused at IND-CPA layer
        L_prime = h_plus[32:]

        # Re-encrypt with derived randomness
        try:
            u_prime, v_prime = self._encrypt_raw(public_key, p_prime, L_prime)
        except Exception as exc:
            # Re-encryption itself failed — treat as mismatch
            garbage = hashlib.sha3_256(self._z + ct_bytes).digest()
            raise ImplicitRejectionError(garbage_key=garbage) from exc

        # Constant-time comparison
        u_match = _constant_time_compare_vec(u, u_prime)
        v_match = _constant_time_compare_poly(v, v_prime)

        if not (u_match and v_match):
            # Collect diagnostic info (non-secret; positions of first mismatch)
            bad_pos = []
            for i, (a, b) in enumerate(zip(v, v_prime)):
                if (a % self.alu.Q) != (b % self.alu.Q):
                    bad_pos.append(i)
                    if len(bad_pos) >= 8:
                        break

            # Raise FOTransformError for internal logging, then wrap
            fo_err = FOTransformError(
                u_mismatch=not u_match,
                v_mismatch=not v_match,
                bad_positions=bad_pos,
            )

            # Implicit rejection: deterministic garbage key
            garbage = hashlib.sha3_256(self._z + ct_bytes).digest()
            raise ImplicitRejectionError(garbage_key=garbage) from fo_err

    # ── noise accumulation helpers ────────────────────────────────────────────

    def _accumulate_noise(self, w: Poly, m_poly: Poly) -> None:
        """
        Record centred noise samples from (w - m_poly) for dynamic dv tuning.
        Keeps at most _noise_window samples to bound memory.
        """
        q = self.alu.Q
        for i in range(min(len(w), len(m_poly))):
            raw = (int(w[i]) - int(m_poly[i])) % q
            # Centre mod q
            if raw > q // 2:
                raw -= q
            self._noise_samples.append(raw)
        if len(self._noise_samples) > self._noise_window:
            self._noise_samples = self._noise_samples[-self._noise_window:]

    # ── single-chunk encrypt ──────────────────────────────────────────────────

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

            if not isinstance(message, (bytes, bytearray)):
                raise InvalidMessageTypeError(type(message))
            if len(message) == 0:
                raise EmptyMessageError()
            if len(message) != self.serializer.chunk_size:
                raise MessageSizeError(len(message), self.serializer.chunk_size)

            if r_seed is None:
                r_seed = os.urandom(32)

            u, v = self._encrypt_raw(public_key, message, r_seed)
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

    # ── single-chunk decrypt (hard, with FO check) ────────────────────────────

    def decrypt(
        self,
        secret_key: Vec,
        ciphertext,
        public_key: Optional[tuple] = None,
        fo_override: Optional[bool] = None,
    ) -> bytes:
        """
        Decrypt a single ciphertext (raw tuple or compressed bytes).

        Parameters
        ----------
        secret_key  : list of k polynomials (from keygen)
        ciphertext  : (u, v) tuple  OR  compressed bytes blob
        public_key  : required when FO transform is enabled (default: None
                      disables FO even if fo_enabled=True)
        fo_override : explicitly enable/disable FO for this call only

        Returns
        -------
        bytes — exactly chunk_size raw bytes

        Raises
        ------
        ImplicitRejectionError — ciphertext failed FO check (c != c').
            Access .garbage_key for the opaque pseudo-random output.
        InvalidSecretKeyError, InvalidCiphertextError, DecompressionError,
        DecryptionError — structural / format errors.
        """
        do_fo = fo_override if fo_override is not None else self._fo_enabled

        try:
            if not isinstance(secret_key, list) or len(secret_key) != self.k:
                raise InvalidSecretKeyError(
                    f"Expected list of {self.k} polynomials."
                )

            # --- decompress if needed ---
            if isinstance(ciphertext, (bytes, bytearray)):
                ct_bytes_raw = bytes(ciphertext)
                try:
                    ciphertext = self._compressor.decompress(ciphertext)
                except DecompressionError:
                    raise
                except Exception as exc:
                    raise InvalidCiphertextError(
                        f"Cannot decompress ciphertext: {exc}"
                    ) from exc
            else:
                # Serialise to canonical bytes for FO comparison
                if isinstance(ciphertext, tuple) and len(ciphertext) == 2:
                    _u_tmp, _v_tmp = ciphertext
                    if isinstance(_u_tmp, list) and isinstance(_v_tmp, list):
                        ct_bytes_raw = _ct_to_bytes(
                            _u_tmp, _v_tmp, self.serializer, self._du, self._dv
                        )
                    else:
                        ct_bytes_raw = b""
                else:
                    ct_bytes_raw = b""

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

            # --- core decryption ---
            s_ntt = [ntt(si) for si in secret_key]
            u_ntt = [ntt(ui) for ui in u]
            su    = self.vmm.vec_dot(s_ntt, u_ntt, a_in_ntt=True, b_in_ntt=True)
            w     = self.alu.poly_sub(v, su)
            p_prime = self.serializer.decode_poly_to_message(w)

            # --- accumulate noise for dynamic dv ---
            m_poly = self.serializer.encode_message_to_poly(p_prime)
            self._accumulate_noise(w, m_poly)

            # --- FO transform check ---
            if do_fo and public_key is not None:
                # May raise ImplicitRejectionError — caller must handle
                self._fo_check(public_key, p_prime, u, v, ct_bytes_raw)

            return p_prime

        except (
            InvalidSecretKeyError, InvalidCiphertextError,
            DecompressionError, ImplicitRejectionError,
        ):
            raise
        except Exception as exc:
            raise DecryptionError(f"Decryption failed: {exc}") from exc

    # ── single-chunk soft-decision decrypt ───────────────────────────────────

    def decrypt_soft(
        self,
        secret_key: Vec,
        ciphertext,
        public_key: Optional[tuple] = None,
        fo_override: Optional[bool] = None,
        confidence_threshold: float = SerializationUnit.DEFAULT_CONFIDENCE_THRESHOLD,
        vote_ciphertexts: Optional[List] = None,
        raise_on_weak: bool = True,
        max_weak_allowed: int = 0,
    ) -> SoftDecodeResult:
        """
        Soft-decision decrypt: returns a SoftDecodeResult with per-bit
        confidence scores and weak-bit metadata instead of bare bytes.

        Parameters
        ----------
        secret_key           : list of k polynomials
        ciphertext           : (u, v) tuple or compressed bytes
        public_key           : required for FO check
        fo_override          : override engine-level fo_enabled flag
        confidence_threshold : bit confidence floor (0.0–1.0)
        vote_ciphertexts     : additional ciphertexts of the same message
                               for majority voting over weak bits
        raise_on_weak        : raise SoftDecodeError if bits remain unresolved
        max_weak_allowed     : tolerate this many unresolved weak bits silently

        Returns
        -------
        SoftDecodeResult

        Raises
        ------
        SoftDecodeError        — unresolvable weak bits (when raise_on_weak=True)
        ImplicitRejectionError — FO check failed
        InvalidSecretKeyError, InvalidCiphertextError, DecryptionError
        """
        do_fo = fo_override if fo_override is not None else self._fo_enabled

        try:
            if not isinstance(secret_key, list) or len(secret_key) != self.k:
                raise InvalidSecretKeyError(
                    f"Expected list of {self.k} polynomials."
                )

            # Decompress primary
            if isinstance(ciphertext, (bytes, bytearray)):
                ct_bytes_raw = bytes(ciphertext)
                try:
                    ciphertext = self._compressor.decompress(ciphertext)
                except DecompressionError:
                    raise
                except Exception as exc:
                    raise InvalidCiphertextError(
                        f"Cannot decompress ciphertext: {exc}"
                    ) from exc
            else:
                if isinstance(ciphertext, tuple) and len(ciphertext) == 2:
                    _u_tmp, _v_tmp = ciphertext
                    ct_bytes_raw = _ct_to_bytes(
                        _u_tmp, _v_tmp, self.serializer, self._du, self._dv
                    ) if isinstance(_u_tmp, list) else b""
                else:
                    ct_bytes_raw = b""

            if not (isinstance(ciphertext, tuple) and len(ciphertext) == 2):
                raise InvalidCiphertextError("Must be a (u, v) tuple.")

            u, v = ciphertext

            # --- primary decryption polynomial w = v - s·u ---
            s_ntt = [ntt(si) for si in secret_key]
            u_ntt = [ntt(ui) for ui in u]
            su    = self.vmm.vec_dot(s_ntt, u_ntt, a_in_ntt=True, b_in_ntt=True)
            w     = self.alu.poly_sub(v, su)

            # --- additional w polynomials from vote ciphertexts ---
            vote_polys: List[Poly] = []
            if vote_ciphertexts:
                for vct in vote_ciphertexts:
                    try:
                        if isinstance(vct, (bytes, bytearray)):
                            vct = self._compressor.decompress(vct)
                        if isinstance(vct, tuple) and len(vct) == 2:
                            vu2, vv2 = vct
                            vu2_ntt  = [ntt(x) for x in vu2]
                            vsu      = self.vmm.vec_dot(
                                s_ntt, vu2_ntt, a_in_ntt=True, b_in_ntt=True)
                            vw       = self.alu.poly_sub(vv2, vsu)
                            vote_polys.append(vw)
                    except Exception:
                        pass  # ignore malformed vote ciphertexts

            # --- soft decode ---
            result = self.serializer.soft_decode_poly(
                w,
                confidence_threshold=confidence_threshold,
                vote_polys=vote_polys if vote_polys else None,
                raise_on_weak=raise_on_weak,
                max_weak_allowed=max_weak_allowed,
            )

            # --- noise accumulation ---
            m_poly = self.serializer.encode_message_to_poly(result.message)
            self._accumulate_noise(w, m_poly)

            # --- FO check on decoded message ---
            if do_fo and public_key is not None:
                self._fo_check(public_key, result.message, u, v, ct_bytes_raw)

            return result

        except (
            InvalidSecretKeyError, InvalidCiphertextError,
            DecompressionError, ImplicitRejectionError, SoftDecodeError,
        ):
            raise
        except Exception as exc:
            raise DecryptionError(f"Soft decryption failed: {exc}") from exc

    # ── dynamic dv selection ──────────────────────────────────────────────────

    def select_dv(
        self,
        security_level: int = 512,
        bandwidth_profile: str = "medium",
        dv_min: int = 2,
        dv_max: int = 12,
    ) -> Tuple[int, dict]:
        """
        Pick the smallest dv that satisfies the current noise budget, the
        security-level floor, and the bandwidth profile.

        Uses the noise samples accumulated across all previous decrypt() /
        decrypt_soft() calls.  Call after a warmup phase of at least a few
        decryptions for reliable estimates.

        Parameters
        ----------
        security_level    : Kyber variant {512, 768, 1024}
        bandwidth_profile : "high" | "medium" | "low"
        dv_min, dv_max    : search range (inclusive)

        Returns
        -------
        (chosen_dv, stats_dict)

        Raises
        ------
        DynamicDvError — no safe dv exists given current noise observations.
        """
        samples = self._noise_samples or [0]
        return select_dv_dynamic(
            observed_noise_samples=samples,
            q=self.alu.Q,
            n=self.serializer.N,
            security_level=security_level,
            bandwidth_profile=bandwidth_profile,
            dv_min=dv_min,
            dv_max=dv_max,
        )

    # ── high-level chunked API ────────────────────────────────────────────────

    def prepare_payload(self, value: Any) -> List[bytes]:
        """Convert any supported Python value to a list of chunk_size blocks."""
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
                if r_seed is not None else None
            )
            result.append(self.encrypt(public_key, chunk, chunk_seed))
        return result

    def decrypt_chunked(
        self,
        secret_key: Vec,
        ciphertexts: List,
        output_type: str = "bytes",
        public_key: Optional[tuple] = None,
        fo_override: Optional[bool] = None,
    ):
        """
        Decrypt a list of ciphertexts produced by encrypt_chunked().

        Parameters
        ----------
        secret_key  : list of k polynomials
        ciphertexts : list of (u, v) tuples or compressed bytes
        output_type : "bytes" | "str" | "int" | "float" (fallback hint)
        public_key  : required when FO is enabled
        fo_override : per-call FO override

        Returns
        -------
        Decoded value in the requested output_type.

        Raises
        ------
        CiphertextListError    — ciphertexts is not a list
        ImplicitRejectionError — any chunk failed FO check
        """
        if not isinstance(ciphertexts, list):
            raise CiphertextListError(type(ciphertexts))

        raw = b"".join(
            self.decrypt(secret_key, ct,
                         public_key=public_key,
                         fo_override=fo_override)
            for ct in ciphertexts
        )
        return _from_bytes(raw, output_type)