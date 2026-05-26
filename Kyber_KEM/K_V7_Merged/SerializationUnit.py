"""
SerializationUnit.py
────────────────────
Handles data transformation between physical bit busses and polynomial
coefficients.

V7 additions
────────────
  • Soft-decision decoding with per-bit confidence scoring
  • Majority-vote resolution of weak bits
  • SoftDecodeError / WeakBitWarning plumbed through
  • Dynamic dv selection based on observed noise, security level, bandwidth
  • DynamicDvError when no safe level exists

COMPRESSION ALGORITHM  (Kyber spec §2.3)
─────────────────────
    compress(x, d)   = round( x * 2^d / Q ) mod 2^d
    decompress(x, d) = round( x * Q / 2^d )

Coefficient bit-width reduction table (N=256, Q=3329):
    d=10 → 320 bytes  (used for u in Kyber-512/768)
    d=11 → 352 bytes  (used for u in Kyber-1024)
    d=4  → 128 bytes  (used for v in Kyber-512/768, floor value)
    d=5  → 160 bytes  (used for v in Kyber-1024)
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .ArithmeticUnit import Poly
from .KyberExceptions import (
    DynamicDvError,
    ParameterError,
    SoftDecodeError,
    WeakBitWarning,
)
from .Profiler import Profiler

Vec = List[Poly]


# ── Soft-decode result container ─────────────────────────────────────────────

@dataclass
class SoftDecodeResult:
    """
    Carries the decoded message together with per-bit confidence metadata.

    Attributes
    ----------
    message           : bytes        — recovered chunk (chunk_size bytes)
    confidences       : list[float]  — per-bit confidence in [0, 1]
                                        1.0 = maximally confident,
                                        0.0 = completely ambiguous
    weak_bit_positions: list[int]    — bit indices resolved by majority vote
    weak_bit_count    : int          — convenience alias for len(weak_bit_positions)
    min_confidence    : float        — worst-case bit confidence
    avg_confidence    : float        — mean confidence across all N bits
    majority_vote_used: bool         — True if any bit needed majority vote
    """
    message            : bytes
    confidences        : List[float]
    weak_bit_positions : List[int]  = field(default_factory=list)

    @property
    def weak_bit_count(self) -> int:
        return len(self.weak_bit_positions)

    @property
    def min_confidence(self) -> float:
        return min(self.confidences) if self.confidences else 0.0

    @property
    def avg_confidence(self) -> float:
        return (sum(self.confidences) / len(self.confidences)
                if self.confidences else 0.0)

    @property
    def majority_vote_used(self) -> bool:
        return bool(self.weak_bit_positions)


# ── Dynamic dv advisor ───────────────────────────────────────────────────────

# Security level → minimum acceptable dv (hard lower bound).
# Tighter levels need more bits to tolerate wider noise.
_SECURITY_MIN_DV = {
    512:  4,   # Kyber-512   (NIST Level 1)
    768:  4,   # Kyber-768   (NIST Level 3)
    1024: 5,   # Kyber-1024  (NIST Level 5)
}

# Bandwidth profile → preferred byte budget per v polynomial.
# Tighter bandwidth = fewer bits allowed for v.
_BANDWIDTH_MAX_BYTES = {
    "high":   160,   # plenty of bandwidth — up to d=5 for N=256
    "medium": 128,   # standard FPGA AXI stream budget
    "low":     96,   # constrained link — push compression harder
}


def select_dv_dynamic(
    observed_noise_samples: List[int],
    q: int,
    n: int,
    security_level: int = 512,
    bandwidth_profile: str = "medium",
    dv_min: int = 2,
    dv_max: int = 12,
) -> Tuple[int, dict]:
    """
    Choose the smallest dv that keeps all observed noise within the LWE budget
    Q/4, respects the security-level floor, and fits the bandwidth profile.

    Parameters
    ----------
    observed_noise_samples : centred noise values (integers) from recent
                             decrypt calls — used to estimate worst-case noise.
    q, n                   : Kyber ring parameters.
    security_level         : one of {512, 768, 1024} — maps to a min-dv floor.
    bandwidth_profile      : "high" | "medium" | "low" — caps bytes/v-poly.
    dv_min, dv_max         : search range (inclusive).

    Returns
    -------
    (chosen_dv, stats_dict)
        chosen_dv  — selected integer
        stats_dict — diagnostic info (max_noise, budget, bandwidth_cap_bytes, …)

    Raises
    ------
    DynamicDvError  — if no dv in [dv_min, dv_max] satisfies all constraints.
    ParameterError  — if arguments are out of range.
    """
    if security_level not in _SECURITY_MIN_DV:
        raise ParameterError(
            "security_level", security_level,
            f"Must be one of {list(_SECURITY_MIN_DV)}."
        )
    if bandwidth_profile not in _BANDWIDTH_MAX_BYTES:
        raise ParameterError(
            "bandwidth_profile", bandwidth_profile,
            f"Must be one of {list(_BANDWIDTH_MAX_BYTES)}."
        )
    if dv_min < 1 or dv_max > 16 or dv_min > dv_max:
        raise ParameterError(
            "dv_min/dv_max", f"[{dv_min},{dv_max}]",
            "Must satisfy 1 ≤ dv_min ≤ dv_max ≤ 16."
        )

    noise_budget        = q // 4
    security_floor      = _SECURITY_MIN_DV[security_level]
    bandwidth_cap_bytes = _BANDWIDTH_MAX_BYTES[bandwidth_profile]

    # Decompress round-trip adds up to  q / 2^(dv+1)  of extra quantisation
    # noise on top of the LWE noise.  We need:
    #   max_observed_noise + q / 2^(dv+1)  <  q / 4
    max_noise = max((abs(x) for x in observed_noise_samples), default=0)

    effective_min = max(dv_min, security_floor)

    chosen_dv = None
    for dv in range(effective_min, dv_max + 1):
        quant_noise    = q // (1 << (dv + 1))          # worst-case quantisation
        total_noise    = max_noise + quant_noise
        v_bytes        = (n * dv + 7) // 8
        bandwidth_ok   = v_bytes <= bandwidth_cap_bytes
        noise_ok       = total_noise < noise_budget
        if noise_ok and bandwidth_ok:
            chosen_dv = dv
            break

    stats = {
        "max_observed_noise" : max_noise,
        "noise_budget"       : noise_budget,
        "security_floor_dv"  : security_floor,
        "bandwidth_cap_bytes": bandwidth_cap_bytes,
        "search_range"       : (effective_min, dv_max),
        "chosen_dv"          : chosen_dv,
    }

    if chosen_dv is None:
        raise DynamicDvError(
            observed_max_noise=max_noise,
            noise_budget=noise_budget,
            min_dv_tried=effective_min,
            max_dv_tried=dv_max,
        )

    return chosen_dv, stats


# ── Main serialisation class ─────────────────────────────────────────────────

class SerializationUnit:
    """
    Handles data transformation between physical bit busses and polynomial
    coefficients.

    Soft-decision decoding (V7)
    ───────────────────────────
    Instead of immediately quantising each coefficient to a hard 0/1, we
    compute two confidence values:

        confidence_1 = |c - Q/2|   (how far c is from the "1" attractor)
        confidence_0 = min(c, Q-c) (how far c is from the "0" attractor)

    A bit is CONFIDENT when max(conf_1, conf_0) / (Q/2) ≥ threshold.
    WEAK bits below threshold are collected for majority voting: if we have
    multiple decryptions of the same chunk (via retries), we take a vote.
    If still unresolved, SoftDecodeError is raised.
    """

    # Default soft-decode threshold: a bit must use at least 20 % of the Q/2
    # range before we trust the hard decision without a majority vote.
    DEFAULT_CONFIDENCE_THRESHOLD = 0.20

    def __init__(self, n: int, q: int):
        self.N          = n
        self.Q          = q
        self.chunk_size = (n + 7) // 8
        self._half_q    = q // 2

    # ── Message encoding / decoding (hard) ──────────────────────────────────

    @Profiler.profile("encoder msg to poly")
    def encode_message_to_poly(self, m: bytes) -> Poly:
        assert len(m) == self.chunk_size, \
            f"Hardware expects exact byte payload interface alignment."
        half_q = self._half_q
        coeffs = []
        for byte in m:
            for bit in range(8):
                if len(coeffs) < self.N:
                    coeffs.append(half_q if (byte >> bit) & 1 else 0)
        return coeffs

    @Profiler.profile("decoder poly to msg (hard)")
    def decode_poly_to_message(self, w: Poly) -> bytes:
        """Original hard-decision decoder.  Fast; no confidence metadata."""
        half_q = self._half_q
        bits = []
        for coef in w:
            c         = coef % self.Q
            dist0     = min(c, self.Q - c)
            dist_half = abs(c - half_q)
            bits.append(1 if dist_half < dist0 else 0)
        result = bytearray(self.chunk_size)
        for i, bit in enumerate(bits):
            result[i // 8] |= bit << (i % 8)
        return bytes(result)

    # ── Soft-decision decoding ───────────────────────────────────────────────

    @Profiler.profile("decoder poly to msg (soft)")
    def soft_decode_poly(
        self,
        w: Poly,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        vote_polys: Optional[List[Poly]] = None,
        raise_on_weak: bool = True,
        max_weak_allowed: int = 0,
    ) -> SoftDecodeResult:
        """
        Soft-decision decode with majority voting for weak bits.

        Parameters
        ----------
        w                    : the noisy decryption polynomial (v - s·u)
        confidence_threshold : bit confidence floor in [0, 1].
                               A bit is "weak" if neither attractor pulls
                               with this fraction of the Q/2 range.
        vote_polys           : additional candidate polys (e.g. from retries)
                               used for majority voting over weak bits.
        raise_on_weak        : if True, raises SoftDecodeError when
                               weak bits remain after voting.
        max_weak_allowed     : how many weak bits to tolerate silently
                               (only meaningful when raise_on_weak=False).

        Returns
        -------
        SoftDecodeResult with .message, .confidences, .weak_bit_positions

        Raises
        ------
        SoftDecodeError  — when unresolvable weak bits exceed max_weak_allowed
                           and raise_on_weak=True.
        """
        q       = self.Q
        half_q  = self._half_q
        n       = self.N
        scale   = half_q  # normalise confidences to [0, 1]

        bits        : List[int]   = []
        confidences : List[float] = []
        weak_idx    : List[int]   = []

        for i, coef in enumerate(w):
            c = coef % q

            # Raw distances to the two attractors (0 and Q/2)
            confidence_1 = abs(c - half_q)   # distance FROM "1"
            confidence_0 = min(c, q - c)      # distance FROM "0"

            # Which attractor wins?
            if confidence_1 < confidence_0:
                # c is closer to Q/2  → bit = 1
                hard_bit   = 1
                confidence = confidence_0 / scale   # margin over the "0" attractor
            else:
                # c is closer to 0   → bit = 0
                hard_bit   = 0
                confidence = confidence_1 / scale   # margin over the "1" attractor

            # Clamp to [0, 1]
            confidence = min(1.0, max(0.0, confidence))

            bits.append(hard_bit)
            confidences.append(confidence)
            if confidence < confidence_threshold:
                weak_idx.append(i)

        # ── majority voting over weak bits ───────────────────────────────────
        resolved_by_vote: List[int] = []

        if weak_idx and vote_polys:
            for bit_i in weak_idx:
                votes = [bits[bit_i]]   # start with primary
                for vp in vote_polys:
                    if bit_i < len(vp):
                        vc = vp[bit_i] % q
                        votes.append(1 if abs(vc - half_q) < min(vc, q - vc) else 0)
                majority = 1 if sum(votes) > len(votes) // 2 else 0
                if majority != bits[bit_i]:
                    bits[bit_i] = majority
                resolved_by_vote.append(bit_i)

            # After voting, recompute which bits are still truly weak
            still_weak = [
                i for i in weak_idx
                if i not in resolved_by_vote and confidences[i] < confidence_threshold
            ]
        else:
            still_weak = weak_idx

        # ── emit WeakBitWarning for bits resolved by majority vote ───────────
        if resolved_by_vote:
            weak_confidences = [confidences[i] for i in resolved_by_vote]
            warnings.warn(WeakBitWarning(resolved_by_vote, weak_confidences))

        # ── raise if irresolvable weak bits remain ───────────────────────────
        if still_weak and raise_on_weak and len(still_weak) > max_weak_allowed:
            raise SoftDecodeError(
                weak_bit_count=len(still_weak),
                threshold=confidence_threshold,
                attempt=1,
            )

        # ── pack bits into bytes ─────────────────────────────────────────────
        result = bytearray(self.chunk_size)
        for i, bit in enumerate(bits[:n]):
            result[i // 8] |= bit << (i % 8)

        return SoftDecodeResult(
            message=bytes(result),
            confidences=confidences,
            weak_bit_positions=resolved_by_vote,
        )

    # ── Core compression primitives ──────────────────────────────────────────

    def _compress_coeff(self, x: int, d: int) -> int:
        x = x % self.Q
        return ((x * (1 << d) + self.Q // 2) // self.Q) % (1 << d)

    def _decompress_coeff(self, x: int, d: int) -> int:
        return (x * self.Q + (1 << (d - 1))) // (1 << d)

    # ── Polynomial-level compress / decompress ───────────────────────────────

    def compress_poly(self, poly: Poly, d: int) -> bytes:
        """Pack each coefficient into d bits (little-endian bit order)."""
        bits = []
        for coef in poly:
            c = self._compress_coeff(coef, d)
            for bit_pos in range(d):
                bits.append((c >> bit_pos) & 1)
        num_bytes = (len(bits) + 7) // 8
        out = bytearray(num_bytes)
        for i, b in enumerate(bits):
            out[i // 8] |= b << (i % 8)
        return bytes(out)

    def decompress_poly(self, data: bytes, d: int) -> Poly:
        """Unpack d-bit coefficients from a byte string."""
        bits = []
        for byte in data:
            for bit_pos in range(8):
                bits.append((byte >> bit_pos) & 1)
        poly = []
        for i in range(self.N):
            c = 0
            for bit_pos in range(d):
                c |= bits[i * d + bit_pos] << bit_pos
            poly.append(self._decompress_coeff(c, d))
        return poly

    # ── Full ciphertext compress / decompress ────────────────────────────────

    def compress_ciphertext(self, u: Vec, v: Poly,
                            du: int = 10, dv: int = 4) -> bytes:
        """Serialise (u, v) to bytes using du bits per u-coeff, dv per v-coeff."""
        out = bytearray()
        for poly in u:
            out += self.compress_poly(poly, du)
        out += self.compress_poly(v, dv)
        return bytes(out)

    def decompress_ciphertext(self, data: bytes, k: int,
                              du: int = 10, dv: int = 4) -> tuple:
        """Deserialise bytes back to (u, v) polynomials."""
        u_poly_bytes = (self.N * du + 7) // 8
        v_poly_bytes = (self.N * dv + 7) // 8
        u, offset = [], 0
        for _ in range(k):
            u.append(self.decompress_poly(data[offset:offset + u_poly_bytes], du))
            offset += u_poly_bytes
        v = self.decompress_poly(data[offset:offset + v_poly_bytes], dv)
        return u, v