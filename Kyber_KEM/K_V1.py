"""
Kyber (ML-KEM) Implementation in Pure Python
Based on the Module Learning With Errors (M-LWE) problem.
Parameters: Kyber-512 (k=2), Kyber-768 (k=3), Kyber-1024 (k=4)

Ring: R_q = Z_q[X] / (X^256 + 1),  q = 3329, n = 256
"""

import os
import hashlib
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# Global Parameters
# ─────────────────────────────────────────────────────────────────────────────

N = 24  # polynomial degree
Q = 31  # modulus
ETA = 2  # binomial distribution parameter (coefficients in {-2,-1,0,1,2})

# NTT root of unity: zeta^512 ≡ 1 (mod 3329),  zeta = 17
ZETA = 8

# ─────────────────────────────────────────────────────────────────────────────
# Helper: modular arithmetic
# ─────────────────────────────────────────────────────────────────────────────


def centered_mod(a: int, q: int = Q) -> int:
    """Reduce a into (-q/2, q/2]."""
    r = a % q
    if r > q // 2:
        r -= q
    return r


def mod_q(a: int) -> int:
    return a % Q


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial in R_q = Z_q[X]/(X^256+1)
# Represented as a list of N integers mod q.
# ─────────────────────────────────────────────────────────────────────────────

Poly = List[int]  # length-N list of ints


def poly_zero() -> Poly:
    return [0] * N


def poly_add(a: Poly, b: Poly) -> Poly:
    return [(a[i] + b[i]) % Q for i in range(N)]


def poly_sub(a: Poly, b: Poly) -> Poly:
    return [(a[i] - b[i]) % Q for i in range(N)]


def poly_neg(a: Poly) -> Poly:
    return [(-a[i]) % Q for i in range(N)]


def poly_scalar_mul(a: Poly, s: int) -> Poly:
    return [(a[i] * s) % Q for i in range(N)]


def poly_mul(a: Poly, b: Poly) -> Poly:
    """
    Schoolbook multiplication mod (X^256 + 1) and mod q.
    Since X^256 ≡ -1, terms of degree >= 256 wrap with a sign flip.
    """
    result = [0] * N
    for i in range(N):
        if a[i] == 0:
            continue
        for j in range(N):
            if b[j] == 0:
                continue
            deg = i + j
            coef = a[i] * b[j]
            if deg < N:
                result[deg] = (result[deg] + coef) % Q
            else:
                # X^(deg) = X^(deg-256) * X^256 ≡ -X^(deg-256)
                result[deg - N] = (result[deg - N] - coef) % Q
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────


def _prf(seed: bytes, nonce: int, length: int) -> bytes:
    """Pseudo-random function: SHAKE-256 with (seed || nonce)."""
    h = hashlib.shake_256()
    h.update(seed + bytes([nonce]))
    return h.digest(length)


def sample_uniform(seed: bytes, i: int, j: int) -> Poly:
    """
    Sample a uniform polynomial from R_q using rejection sampling (XOF).
    Deterministically from (seed, i, j) — used to build matrix A.
    """
    h = hashlib.shake_128()
    h.update(seed + bytes([i, j]))
    stream = h.digest(3 * N * 2)  # over-generate; rejection sampling
    result = []
    idx = 0
    while len(result) < N:
        if idx + 2 >= len(stream):  # refill if exhausted
            h2 = hashlib.shake_128()
            h2.update(stream[-32:])
            stream += h2.digest(3 * N)
        b0, b1, b2 = stream[idx], stream[idx + 1], stream[idx + 2]
        idx += 3
        d1 = b0 + 256 * (b1 & 0x0F)
        d2 = (b1 >> 4) + 16 * b2
        if d1 < Q:
            result.append(d1)
        if d2 < Q and len(result) < N:
            result.append(d2)
    return result[:N]


def sample_cbd(seed: bytes, nonce: int, eta: int = ETA) -> Poly:
    """
    Sample a polynomial from Centered Binomial Distribution CBD(eta).
    Each coefficient = sum of eta bits - sum of eta bits ∈ {-eta,...,eta}.
    """
    num_bytes = 64 * eta
    buf = _prf(seed, nonce, num_bytes)
    bits = []
    for byte in buf:
        for bit_pos in range(8):
            bits.append((byte >> bit_pos) & 1)
    coeffs = []
    for i in range(N):
        a = sum(bits[2 * eta * i + j] for j in range(eta))
        b = sum(bits[2 * eta * i + eta + j] for j in range(eta))
        coeffs.append((a - b) % Q)
    return coeffs


# ─────────────────────────────────────────────────────────────────────────────
# Module (vector / matrix of polynomials)
# ─────────────────────────────────────────────────────────────────────────────

Vec = List[Poly]  # k polynomials
Mat = List[Vec]  # k x k polynomials


def vec_add(a: Vec, b: Vec) -> Vec:
    return [poly_add(a[i], b[i]) for i in range(len(a))]


def vec_dot(a: Vec, b: Vec) -> Poly:
    """Inner product of two polynomial vectors → single polynomial."""
    acc = poly_zero()
    for ai, bi in zip(a, b):
        acc = poly_add(acc, poly_mul(ai, bi))
    return acc


def mat_vec_mul(A: Mat, v: Vec) -> Vec:
    """Matrix-vector product: A * v."""
    k = len(A)
    return [vec_dot(A[i], v) for i in range(k)]


def mat_transpose(A: Mat) -> Mat:
    k = len(A)
    return [[A[j][i] for j in range(k)] for i in range(k)]


def generate_A(rho: bytes, k: int) -> Mat:
    """Generate the public matrix
    A ∈ R_q^{k×k}
    deterministically from seed rho."""
    return [[sample_uniform(rho, i, j) for j in range(k)] for i in range(k)]


# ─────────────────────────────────────────────────────────────────────────────
# Message encode / decode
# ─────────────────────────────────────────────────────────────────────────────


def encode_message(m: bytes) -> Poly:
    """
    Map 32-byte (256-bit) message to polynomial.
    Bit 1 → coefficient ⌊q/2⌋ = 1664  (closest integer to q/2)
    Bit 0 → coefficient 0
    """
    assert len(m) == 32
    half_q = Q // 2  # 1664
    coeffs = []
    for byte in m:
        for bit in range(8):
            coeffs.append(half_q if (byte >> bit) & 1 else 0)
    return coeffs  # length 256


def decode_message(w: Poly) -> bytes:
    """
    Decode polynomial back to 32-byte message.
    Coefficient closer to ⌊q/2⌋ → bit 1, closer to 0 → bit 0.
    """
    half_q = Q // 2
    bits = []
    for coef in w:
        c = coef % Q
        # Distance to 0 vs distance to half_q
        dist0 = min(c, Q - c)
        dist_half = abs(c - half_q)
        bits.append(1 if dist_half < dist0 else 0)
    # Pack bits into bytes (LSB first)
    result = bytearray(32)
    for i, bit in enumerate(bits):
        result[i // 8] |= bit << (i % 8)
    return bytes(result)


# ─────────────────────────────────────────────────────────────────────────────
# Kyber Core: Key Generation, Encrypt, Decrypt
# ─────────────────────────────────────────────────────────────────────────────


def keygen(k: int = 2, seed: bytes = None):
    """
    Key Generation.
    Returns:
        public_key  = (A, t)   where t = A*s + e  (mod q)
        secret_key  = s
    """
    if seed is None:
        seed = os.urandom(32)

    # Expand seed into two 32-byte seeds
    h = hashlib.sha3_512(seed).digest()
    rho = h[:32]  # seed for A
    sigma = h[32:]  # seed for s, e

    A = generate_A(rho, k)

    # Sample secret and error vectors from CBD
    nonce = 0
    s: Vec = []
    for _ in range(k):
        s.append(sample_cbd(sigma, nonce))
        nonce += 1

    e: Vec = []
    for _ in range(k):
        e.append(sample_cbd(sigma, nonce))
        nonce += 1

    # t = A*s + e  (mod q)
    As = mat_vec_mul(A, s)
    t = vec_add(As, e)

    public_key = (rho, t)  # send rho instead of full A (receiver re-derives A)
    secret_key = s
    return public_key, secret_key


def encrypt(public_key, message: bytes, k: int = 2, r_seed: bytes = None):
    """
    Encryption.
      u = A^T * r + e1   (mod q)
      v = t^T * r + e2 + m_encoded  (mod q)
    Returns ciphertext (u, v).
    """
    assert len(message) == 32, "Message must be exactly 32 bytes (256 bits)"
    rho, t = public_key

    if r_seed is None:
        r_seed = os.urandom(32)

    A = generate_A(rho, k)
    AT = mat_transpose(A)

    nonce = 0
    r: Vec = []
    for _ in range(k):
        r.append(sample_cbd(r_seed, nonce))
        nonce += 1

    e1: Vec = []
    for _ in range(k):
        e1.append(sample_cbd(r_seed, nonce))
        nonce += 1

    e2: Poly = sample_cbd(r_seed, nonce)

    # u = A^T * r + e1
    ATr = mat_vec_mul(AT, r)
    u = vec_add(ATr, e1)

    # v = t^T * r + e2 + m_bar
    tTr = vec_dot(t, r)
    m_bar = encode_message(message)
    v = poly_add(poly_add(tTr, e2), m_bar)

    return (u, v)


def decrypt(secret_key, ciphertext, k: int = 2) -> bytes:
    """
    Decryption.
      w = v - s^T * u   (mod q)
        = m_bar + noise      (the noise is small enough to round away)
    """
    s = secret_key
    u, v = ciphertext

    # w = v - s^T * u
    sTu = vec_dot(s, u)
    w = poly_sub(v, sTu)

    return decode_message(w)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────


class Kyber:
    """High-level Kyber interface. Choose variant: 512, 768, or 1024."""

    VARIANTS : dict = {512: 2, 768: 3, 1024: 4}
    self.to_send = msg if msg is not None else self.to_send
    to_send : bytes = None

    def __init__(self, variant: int = 512):
        assert variant in self.VARIANTS, f"variant must be one of {list(self.VARIANTS)}"
        self.k = self.VARIANTS[variant]
        self.variant = variant

    def prepare_message(self, message: str): 
        byte_data : bytes = message.encode('utf-8')
        self.to_send = byte_data[:32].ljust(32, b'\x00')
        
    def keygen(self, seed: bytes = None):
        return keygen(self.k, seed)

    def encrypt(self, public_key, msg: bytes = None,  r_seed: bytes = None):
        self.to_send = msg if msg is not None else self.to_send
        return encrypt(public_key, self.to_send, self.k, r_seed)

    def decrypt(self, secret_key, ciphertext) -> bytes:
        return decrypt(secret_key, ciphertext, self.k)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  Kyber (M-LWE) Pure-Python Implementation")
    print("=" * 60)

    for variant in [512, 768, 1024]:
        print(f"\n── Kyber-{variant} (k={Kyber.VARIANTS[variant]}) ──────────────")
        kyber = Kyber(variant)

        # Key generation
        t0 = time.perf_counter()
        pk, sk = kyber.keygen()
        t_keygen = time.perf_counter() - t0

        # Encrypt a random message
        message = "Hello World!"
        kyber.prepare_message(message)
        
        # Capture the actual processed bytes to check against decryption
        expected_bytes = kyber.to_send 
        
        t0 = time.perf_counter()
        ct = kyber.encrypt(pk)
        t_enc = time.perf_counter() - t0

        # Decrypt
        t0 = time.perf_counter()
        recovered = kyber.decrypt(sk, ct)
        t_dec = time.perf_counter() - t0

        # Compare bytes to bytes instead of bytes to string
        ok = "✓ PASS" if recovered == expected_bytes else "✗ FAIL"

        print(f"  Decryption:  {ok}")
        print(f"  Message (str):     {message[:32]}...")
        print(f"  Message (bytes):     {expected_bytes.hex()[:32]}...")
        print(f"  Recovered (bytes):   {recovered.hex()[:32]}...")
        print(f"  Recovered (str):   {recovered.decode('utf-8')[:32]}...")
        print(f"  KeyGen:      {t_keygen*1000:.1f} ms")
        print(f"  Encrypt:     {t_enc*1000:.1f} ms")
        print(f"  Decrypt:     {t_dec*1000:.1f} ms")

    print("\n" + "=" * 60)
    print("  Noise Analysis Demo (Kyber-512)")
    print("=" * 60)
    kyber = Kyber(512)
    pk, sk = kyber.keygen()
    msg = os.urandom(N // 8)
    ct = kyber.encrypt(pk, msg)
    u, v = ct
    s = sk
    sTu = vec_dot(s, u)
    w = poly_sub(v, sTu)
    m_bar = encode_message(msg)

    noise = [(w[i] - m_bar[i]) % Q for i in range(N)]
    noise_centered = [centered_mod(n) for n in noise]
    max_noise = max(abs(n) for n in noise_centered)
    print(f"  q/4 bound:       {Q//4}")
    print(
        f"  Max |E_i|:       {max_noise}  ({'within bound' if max_noise < Q//4 else 'EXCEEDED — decryption failure'})"
    )
    print(f"  Noise sample:    {noise_centered[:10]}")
    print()
