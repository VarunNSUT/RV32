from typing import List
from .Profiler import Profiler

Poly = List[int]

# ─────────────────────────────────────────────────────────────────────────────
# Kyber NTT constants  (FIPS 203 §4.3)
# Ring: Z_3329[X]/(X^256 + 1)  — negacyclic NTT of length 256
# ζ = 17,  zetas[k] = 17^{bitrev_7(k)} mod 3329
# ─────────────────────────────────────────────────────────────────────────────

_Q = 3329
_N = 256


def _bitrev7(n: int) -> int:
    r = 0
    for _ in range(7):
        r = (r << 1) | (n & 1)
        n >>= 1
    return r


ZETA = [pow(17, _bitrev7(k), _Q) for k in range(128)]
_F_INV = pow(128, _Q - 2, _Q)   # 128^{-1} mod 3329 = 3303

@Profiler.profile("Forward NTT")
def ntt(f: Poly) -> Poly:
    """Cooley-Tukey NTT (Algorithm 9, FIPS 203). O(N log N)."""
    a = list(f)
    k = 1
    l = 128
    while l >= 2:
        for start in range(0, 256, 2 * l):
            zeta = ZETA[k]; k += 1
            for j in range(start, start + l):
                t = zeta * a[j + l] % _Q
                a[j + l] = (a[j] - t) % _Q
                a[j]     = (a[j] + t) % _Q
        l >>= 1
    return a

@Profiler.profile("Inverse NTT")
def intt(f: Poly) -> Poly:
    """Gentleman-Sande INTT (Algorithm 10, FIPS 203)."""
    a = list(f)
    k = 127
    l = 2
    while l <= 128:
        for start in range(0, 256, 2 * l):
            zeta = ZETA[k]; k -= 1
            for j in range(start, start + l):
                t        = a[j]
                a[j]     = (t + a[j + l]) % _Q
                a[j + l] = zeta * (a[j + l] - t) % _Q
        l <<= 1
    return [c * _F_INV % _Q for c in a]


@Profiler.profile("Multiplication NTT")
def ntt_mul(a_hat: Poly, b_hat: Poly) -> Poly:
    """
    Base-case multiplication in NTT domain (Algorithm 11, FIPS 203).
    Decomposes into 64 degree-2 products with ±zeta[64+i] factors.
    """
    c = [0] * 256
    for i in range(64):
        a0, a1 = a_hat[4*i],   a_hat[4*i+1]
        a2, a3 = a_hat[4*i+2], a_hat[4*i+3]
        b0, b1 = b_hat[4*i],   b_hat[4*i+1]
        b2, b3 = b_hat[4*i+2], b_hat[4*i+3]
        z = ZETA[64 + i]
        c[4*i]   = (a0*b0 + z        * a1*b1) % _Q
        c[4*i+1] = (a0*b1 + a1*b0)            % _Q
        c[4*i+2] = (a2*b2 + (_Q - z) * a3*b3) % _Q
        c[4*i+3] = (a2*b3 + a3*b2)            % _Q
    return c


class ArithmeticUnit:
    """
    Maps directly to physical DSP slices and combinational logic paths.
    NTT multiplication uses Kyber-spec NTT — O(N log N) vs O(N²) schoolbook.
    """

    def __init__(self, n: int, q: int):
        self.N = n
        self.Q = q

    def poly_add(self, a: Poly, b: Poly) -> Poly:
        return [(a[i] + b[i]) % self.Q for i in range(self.N)]

    def poly_sub(self, a: Poly, b: Poly) -> Poly:
        return [(a[i] - b[i]) % self.Q for i in range(self.N)]

    def poly_scalar_mul(self, a: Poly, s: int) -> Poly:
        return [(a[i] * s) % self.Q for i in range(self.N)]

    def poly_zero(self) -> Poly:
        return [0] * self.N

    @Profiler.profile("ALU: Polynomial Multiply (ntt)")
    def poly_mul_ntt(self, a: Poly, b: Poly) -> Poly:
        """NTT-based ring multiplication. O(N log N)."""
        return intt(ntt_mul(ntt(a), ntt(b)))
    
    @Profiler.profile("ALU: Polynomial Multiply (ntt domain)")
    def poly_mul_ntt_domain(self, a_ntt: Poly, b_ntt: Poly) -> Poly:
        """Multiply when BOTH inputs are already in NTT domain."""
        return intt(ntt_mul(a_ntt, b_ntt))

    @Profiler.profile("ALU: Polynomial Multiply (Schoolbook)")
    def poly_mul_schoolbook(self, a: Poly, b: Poly) -> Poly:
        """
        O(N²) schoolbook — kept for reference/testing.
        Warning: In hardware, this maps to N^2 physical multipliers if unrolled.
        """
        result = [0] * self.N
        for i in range(self.N):
            if a[i] == 0:
                continue
            for j in range(self.N):
                if b[j] == 0:
                    continue
                deg  = i + j
                coef = a[i] * b[j]
                if deg < self.N:
                    result[deg] = (result[deg] + coef) % self.Q
                else:
                    result[deg - self.N] = (result[deg - self.N] - coef) % self.Q
        return result

    def centered_mod(self, a: int) -> int:
        r = a % self.Q
        if r > self.Q // 2:
            r -= self.Q
        return r
























