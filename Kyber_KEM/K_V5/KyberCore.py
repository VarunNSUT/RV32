import hashlib
from .ArithmeticUnit import ArithmeticUnit, ntt, intt, ntt_mul
from .SamplingUnit import SamplingUnit
from .VectorMatrixMultiplier import VectorMatrixMultiplier
from .SerializationUnit import SerializationUnit


class KyberCoreEngine:
    """
    Central Hardware FSM scheduling operations.

    Optimisations
    ─────────────
    1. NTT multiplication   — O(N log N) vs O(N²) schoolbook
    2. A stored in NTT domain — generate_A_matrix emits NTT polys directly
    3. A_cache keyed on rho — identical public key reuses the same A matrix
    4. r and s NTT'd once   — reused across all dot-products in encrypt/decrypt

    Kyber variant defaults:
        Kyber-512  : k=2, du=10, dv=4  → 768  byte ciphertext
        Kyber-768  : k=3, du=10, dv=4  → 1088 byte ciphertext
        Kyber-1024 : k=4, du=11, dv=5  → 1568 byte ciphertext
    """

    def __init__(self, n: int, q: int, eta: int, k_dimension: int):
        self.k          = k_dimension
        self.alu        = ArithmeticUnit(n, q)
        self.sampler    = SamplingUnit(n, q, eta)
        self.vmm        = VectorMatrixMultiplier(self.alu)
        self.serializer = SerializationUnit(n, q)
        self._du = 10
        self._dv = 4
        self.A_cache: dict = {}   # rho → NTT-domain A matrix

    def _get_A(self, rho: bytes) -> list:
        """Return cached NTT-domain A matrix; generate only once per rho."""
        if rho not in self.A_cache:
            self.A_cache[rho] = self.vmm.generate_A_matrix(
                self.sampler, rho, self.k, store_ntt=True)
        return self.A_cache[rho]

    def keygen(self, seed: bytes) -> tuple:
        h          = hashlib.sha3_512(seed).digest()
        rho, sigma = h[:32], h[32:]
        A_ntt = self._get_A(rho)

        s     = [self.sampler.sample_cbd(sigma, idx)          for idx in range(self.k)]
        e     = [self.sampler.sample_cbd(sigma, self.k + idx) for idx in range(self.k)]
        s_ntt = [ntt(si) for si in s]

        # t = A·s + e  (A and s both NTT-domain — no redundant fwd NTT)
        As = self.vmm.mat_vec_mul(A_ntt, s_ntt, A_in_ntt=True, v_in_ntt=True)
        t  = self.vmm.vec_add(As, e)
        return (rho, t), s   # s returned in coefficient domain for decryption

    def encrypt(self, public_key, message: bytes, r_seed: bytes) -> bytes:
        rho, t = public_key
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
        return self.serializer.compress_ciphertext(u, v, du=self._du, dv=self._dv)

    def decrypt(self, secret_key, ciphertext: bytes) -> bytes:
        u, v = self.serializer.decompress_ciphertext(
            ciphertext, self.k, du=self._du, dv=self._dv)
        s_ntt = [ntt(si) for si in secret_key]
        u_ntt = [ntt(ui) for ui in u]
        su    = self.vmm.vec_dot(s_ntt, u_ntt, a_in_ntt=True, b_in_ntt=True)
        w     = self.alu.poly_sub(v, su)
        return self.serializer.decode_poly_to_message(w)