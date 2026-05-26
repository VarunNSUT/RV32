from typing import List
from .ArithmeticUnit import Poly
from .Profiler import Profiler

Vec = List[Poly]


class SerializationUnit:
    """
    Handles data transformation between physical bit busses and polynomial coefficients.

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

    def __init__(self, n: int, q: int):
        self.N = n
        self.Q = q
        self.chunk_size = (n + 7) // 8

    # ── Message encoding / decoding ──────────────────────────────────────────

    @Profiler.profile("encoder msg to poly")
    def encode_message_to_poly(self, m: bytes) -> Poly:
        assert len(m) == self.chunk_size, \
            f"Hardware expects exact byte payload interface alignment."
        half_q = self.Q // 2
        coeffs = []
        for byte in m:
            for bit in range(8):
                if len(coeffs) < self.N:
                    coeffs.append(half_q if (byte >> bit) & 1 else 0)
        return coeffs

    @Profiler.profile("decoder poly to msg")
    def decode_poly_to_message(self, w: Poly) -> bytes:
        half_q = self.Q // 2
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
