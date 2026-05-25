"""
Kyber (ML-KEM) Implementation in Pure Python
Fully Parameterized for custom N and Q values.
"""
# heheheheheh
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
        self.chunk_size = (N + 7) // 8   # bytes per Kyber block (32 for N=256)
        self._chunks: List[bytes] = []   # prepared chunks ready for encryption

    # ── message preparation ──────────────────────────────────────────────────

    def prepare_message(self, message: str) -> None:
        """
        UTF-8 encode the message and split it into chunk_size blocks,
        padding the final block with null bytes if needed.
        Works for messages of any length.
        """
        raw = message.encode("utf-8")
        self._chunks = []
        for offset in range(0, max(len(raw), 1), self.chunk_size):
            block = raw[offset : offset + self.chunk_size]
            if len(block) < self.chunk_size:
                block = block.ljust(self.chunk_size, b"\x00")   # pad last chunk
            self._chunks.append(block)

    # ── key generation ───────────────────────────────────────────────────────

    def keygen(self, seed: bytes = None):
        return keygen(self.k, seed)

    # ── chunked encrypt ──────────────────────────────────────────────────────

    def encrypt(self, public_key, msg: bytes = None, r_seed: bytes = None):
        """
        Encrypt all chunks prepared by prepare_message() (or a single raw
        bytes block passed via `msg`).  Returns a list of ciphertexts — one
        per chunk — so any caller can treat it as a transparent list.
        """
        if msg is not None:
            # Caller handed us a raw bytes block: treat as a single chunk
            assert len(msg) == self.chunk_size, (
                f"Raw msg must be exactly {self.chunk_size} bytes; "
                f"use prepare_message() for arbitrary strings."
            )
            chunks = [msg]
        else:
            assert self._chunks, "Call prepare_message() before encrypt()."
            chunks = self._chunks

        ciphertexts = []
        for i, chunk in enumerate(chunks):
            # Give each chunk a distinct ephemeral seed so noise is independent
            chunk_r_seed = (
                hashlib.sha256((r_seed or b"") + i.to_bytes(4, "big")).digest()
                if r_seed is not None
                else None
            )
            ciphertexts.append(encrypt(public_key, chunk, self.k, chunk_r_seed))
        return ciphertexts

    # ── chunked decrypt ──────────────────────────────────────────────────────

    def decrypt(self, secret_key, ciphertexts) -> str:
        """
        Decrypt a list of ciphertexts (as returned by encrypt()), reassemble
        the chunks, strip trailing null-byte padding, and return the original
        string.
        """
        raw = b"".join(decrypt(secret_key, ct, self.k) for ct in ciphertexts)
        return raw.rstrip(b"\x00").decode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Execution Runtime Loop
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # ── Test messages: short, exact-fit, and long ────────────────────────────
    TEST_MESSAGES = {
        "short"    : "Hi!",
        "exact"    : "A" * 32,          # exactly one chunk for N=256
        "long"     : (
            "Hello World! Standard cryptographic text verification data. "
            "This message is intentionally long to exercise the chunking path. "
            "Kyber encrypts it block by block and recovers every character."
        ),
    }

    print("=" * 65)
    print(f"  Kyber Custom Instance Configuration: N={N}, Q={Q}")
    print("=" * 65)

    for variant in [512, 768, 1024]:
        print(f"\n── Kyber-{variant} (k={Kyber.VARIANTS[variant]}) " + "─" * 30)
        kyber = Kyber(variant)

        t0 = time.perf_counter()
        pk, sk = kyber.keygen()
        t_keygen = time.perf_counter() - t0

        for label, msg_str in TEST_MESSAGES.items():
            kyber.prepare_message(msg_str)
            num_chunks = len(kyber._chunks)

            t0 = time.perf_counter()
            cts = kyber.encrypt(pk)
            t_enc = time.perf_counter() - t0

            t0 = time.perf_counter()
            recovered_str = kyber.decrypt(sk, cts)
            t_dec = time.perf_counter() - t0

            ok = "✓ PASS" if recovered_str == msg_str else "✗ FAIL"
            print(f"\n  [{label}]  chunks={num_chunks}  {ok}")
            print(f"  Original : {repr(msg_str[:60])}{'...' if len(msg_str) > 60 else ''}")
            print(f"  Recovered: {repr(recovered_str[:60])}{'...' if len(recovered_str) > 60 else ''}")
            print(f"  KeyGen={t_keygen*1000:.1f}ms  Encrypt={t_enc*1000:.1f}ms  Decrypt={t_dec*1000:.1f}ms")

    # ── Noise analysis ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Noise Evaluation Module")
    print("=" * 65)
    kyber = Kyber(512)
    pk, sk = kyber.keygen()
    msg = os.urandom(kyber.chunk_size)
    ct = encrypt(pk, msg, kyber.k)          # single raw call
    u, v = ct
    w = poly_sub(v, vec_dot(sk, u))
    m_bar = encode_message(msg)

    noise = [(w[i] - m_bar[i]) % Q for i in range(N)]
    noise_centered = [centered_mod(n) for n in noise]
    max_noise = max(abs(n) for n in noise_centered)

    print(f"  q/4 error correction ceiling:  {Q // 4}")
    print(f"  Max Absolute Observed Noise:   {max_noise}  "
          f"({'within bound' if max_noise < Q // 4 else 'EXCEEDED — decryption failure'})")
    print(f"  Noise Distribution (first 10): {noise_centered[:10]}")
    print()
