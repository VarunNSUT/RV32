"""
Kyber (ML-KEM) Implementation in Pure Python
Fully Parameterized for custom N and Q values.
"""

import os
import hashlib
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# Global Customizable Parameters
# ─────────────────────────────────────────────────────────────────────────────

N = 256      # Polynomial degree (Can be scaled up to 256 or down to a toy example)
Q = 3329     # Modulus 
ETA = 2    # Binomial distribution parameter

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
# Polynomial in R_q = Z_q[X]/(X^N + 1)
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
    Schoolbook multiplication mod (X^N + 1) and mod q.
    Terms of degree >= N wrap with a sign flip.
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
                # X^(deg) = X^(deg-N) * X^N ≡ -X^(deg-N)
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
    """Sample a uniform polynomial using rejection sampling from seed."""
    h = hashlib.shake_128()
    h.update(seed + bytes([i, j]))
    stream = h.digest(3 * N * 2 + 32)
    result = []
    idx = 0
    while len(result) < N:
        if idx + 2 >= len(stream):
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
    """Sample a polynomial from Centered Binomial Distribution CBD(eta)."""
    num_bytes = ((2 * eta * N) + 7) // 8
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
# Module Mechanics
# ─────────────────────────────────────────────────────────────────────────────

Vec = List[Poly]
Mat = List[Vec]

def vec_add(a: Vec, b: Vec) -> Vec:
    return [poly_add(a[i], b[i]) for i in range(len(a))]

def vec_dot(a: Vec, b: Vec) -> Poly:
    acc = poly_zero()
    for ai, bi in zip(a, b):
        acc = poly_add(acc, poly_mul(ai, bi))
    return acc

def mat_vec_mul(A: Mat, v: Vec) -> Vec:
    k = len(A)
    return [vec_dot(A[i], v) for i in range(k)]

def mat_transpose(A: Mat) -> Mat:
    k = len(A)
    return [[A[j][i] for j in range(k)] for i in range(k)]

def generate_A(rho: bytes, k: int) -> Mat:
    return [[sample_uniform(rho, i, j) for j in range(k)] for i in range(k)]

# ─────────────────────────────────────────────────────────────────────────────
# Customizable Message Encode / Decode
# ─────────────────────────────────────────────────────────────────────────────

def encode_message(m: bytes) -> Poly:
    """
    Map variable-length message to a polynomial of degree N.
    Each coefficient represents exactly 1 bit.
    """
    expected_bytes = (N + 7) // 8
    assert len(m) == expected_bytes, f"Expected exactly {expected_bytes} bytes for N={N}"
    
    half_q = Q // 2
    coeffs = []
    for byte in m:
        for bit in range(8):
            if len(coeffs) < N:
                coeffs.append(half_q if (byte >> bit) & 1 else 0)
    return coeffs

def decode_message(w: Poly) -> bytes:
    """Decode polynomial back to a variable-length byte string."""
    half_q = Q // 2
    bits = []
    for coef in w:
        c = coef % Q
        dist0 = min(c, Q - c)
        dist_half = abs(c - half_q)
        bits.append(1 if dist_half < dist0 else 0)
        
    expected_bytes = (N + 7) // 8
    result = bytearray(expected_bytes)
    for i, bit in enumerate(bits):
        result[i // 8] |= bit << (i % 8)
    return bytes(result)

# ─────────────────────────────────────────────────────────────────────────────
# Kyber Core Encryption Engine
# ─────────────────────────────────────────────────────────────────────────────

def keygen(k: int = 2, seed: bytes = None):
    if seed is None:
        seed = os.urandom(32)
    h = hashlib.sha3_512(seed).digest()
    rho = h[:32]
    sigma = h[32:]

    A = generate_A(rho, k)
    nonce = 0
    s = [sample_cbd(sigma, nonce + idx) for idx in range(k)]
    nonce += k
    e = [sample_cbd(sigma, nonce + idx) for idx in range(k)]

    As = mat_vec_mul(A, s)
    t = vec_add(As, e)
    return (rho, t), s

def encrypt(public_key, message: bytes, k: int = 2, r_seed: bytes = None):
    rho, t = public_key
    if r_seed is None:
        r_seed = os.urandom(32)

    A = generate_A(rho, k)
    AT = mat_transpose(A)

    nonce = 0
    r = [sample_cbd(r_seed, nonce + idx) for idx in range(k)]
    nonce += k
    e1 = [sample_cbd(r_seed, nonce + idx) for idx in range(k)]
    nonce += k
    e2 = sample_cbd(r_seed, nonce)

    ATr = mat_vec_mul(AT, r)
    u = vec_add(ATr, e1)

    tTr = vec_dot(t, r)
    m_bar = encode_message(message)
    v = poly_add(poly_add(tTr, e2), m_bar)
    return (u, v)

def decrypt(secret_key, ciphertext, k: int = 2) -> bytes:
    s = secret_key
    u, v = ciphertext
    sTu = vec_dot(s, u)
    w = poly_sub(v, sTu)
    return decode_message(w)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Kyber:
    VARIANTS = {512: 2, 768: 3, 1024: 4}
    
    def __init__(self, variant: int = 512):
        self.k = self.VARIANTS[variant]
        self.variant = variant
        self.required_bytes = (N + 7) // 8
        self.to_send = b"\x00" * self.required_bytes

    def prepare_message(self, message: str): 
        byte_data = message.encode('utf-8')
        self.to_send = byte_data[:self.required_bytes].ljust(self.required_bytes, b'\x00')
        
    def keygen(self, seed: bytes = None):
        return keygen(self.k, seed)

    def encrypt(self, public_key, msg: bytes = None, r_seed: bytes = None):
        msg_to_encrypt = msg if msg is not None else self.to_send
        return encrypt(public_key, msg_to_encrypt, self.k, r_seed)

    def decrypt(self, secret_key, ciphertext) -> bytes:
        return decrypt(secret_key, ciphertext, self.k)

# ─────────────────────────────────────────────────────────────────────────────
# Execution Runtime Loop
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    print("=" * 60)
    print(f"  Kyber Custom Instance Configuration: N={N}, Q={Q}")
    print("=" * 60)

    for variant in [512, 768, 1024]:
        print(f"\n── Kyber-{variant} (k={Kyber.VARIANTS[variant]}) ──────────────")
        kyber = Kyber(variant)

        t0 = time.perf_counter()
        pk, sk = kyber.keygen()
        t_keygen = time.perf_counter() - t0

        # Generate text payload adjusted dynamically to structural length
        raw_payload = "Hello World! Standard cryptographic text verification data."
        kyber.prepare_message(raw_payload)
        expected_bytes = kyber.to_send 
        
        
        t0 = time.perf_counter()
        ct = kyber.encrypt(pk)
        t_encrypt = time.perf_counter() - t0


        t0 = time.perf_counter()
        recovered = kyber.decrypt(sk, ct)
        t_decrypt = time.perf_counter() - t0

        ok = "✓ PASS" if recovered == expected_bytes else "✗ FAIL"

        print(f"  Decryption:       {ok}")
        print(f"  Message (bytes):   {expected_bytes.hex()}")
        print(f"  Message (str):     {expected_bytes.decode('utf-8')}")
        print(f"  Recovered(bytes):  {recovered.hex()}")
        print(f"  KeyGen:      {t_keygen*1000:.1f} ms")
        print(f"  Encrypt:     {t_encrypt*1000:.1f} ms")
        print(f"  Decrypt:     {t_decrypt*1000:.1f} ms")
        try:
            print(f"  Recovered (str):   {recovered.decode('utf-8', errors='replace')}")
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  Noise Evaluation Module")
    print("=" * 60)
    kyber = Kyber(512)
    pk, sk = kyber.keygen()
    
    # Generate arbitrary validation payload matched to length
    msg = os.urandom(kyber.required_bytes)
    ct = kyber.encrypt(pk, msg)
    u, v = ct
    w = poly_sub(v, vec_dot(sk, u))
    m_bar = encode_message(msg)

    noise = [(w[i] - m_bar[i]) % Q for i in range(N)]
    noise_centered = [centered_mod(n) for n in noise]
    max_noise = max(abs(n) for n in noise_centered)
    
    print(f"  q/4 error correction ceiling:   {Q // 4}")
    print(f"  Max Absolute Observed Noise:    {max_noise}  ({'within bound' if max_noise < Q//4 else 'EXCEEDED — expected decryption failure'})")
    print(f"  Noise Distribution Stream:      {noise_centered}")
    print()
