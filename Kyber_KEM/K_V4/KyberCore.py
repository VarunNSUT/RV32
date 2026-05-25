from .ArithmeticUnit import ArithmeticUnit
from .SamplingUnit import SamplingUnit
from .VectorMatrixMultiplier import VectorMatrixMultiplier 
from .SerializationUnit import SerializationUnit
import hashlib

class KyberCoreEngine:
    """The central Hardware Finite State Machine (FSM) scheduling operations."""
    
    def __init__(self, n: int, q: int, eta: int, k_dimension: int):
        self.k = k_dimension
        self.alu = ArithmeticUnit(n, q)
        self.sampler = SamplingUnit(n, q, eta)
        self.vmm = VectorMatrixMultiplier(self.alu)
        self.serializer = SerializationUnit(n, q)

    def keygen(self, seed: bytes) -> tuple:
        h = hashlib.sha3_512(seed).digest()
        rho, sigma = h[:32], h[32:]

        A = self.vmm.generate_A_matrix(self.sampler, rho, self.k)
        
        s = [self.sampler.sample_cbd(sigma, idx) for idx in range(self.k)]
        e = [self.sampler.sample_cbd(sigma, self.k + idx) for idx in range(self.k)]

        As = self.vmm.mat_vec_mul(A, s)
        t = self.vmm.vec_add(As, e)
        return (rho, t), s

    def encrypt(self, public_key, message: bytes, r_seed: bytes) -> tuple:
        rho, t = public_key
        A = self.vmm.generate_A_matrix(self.sampler, rho, self.k)
        AT = self.vmm.mat_transpose(A)

        r = [self.sampler.sample_cbd(r_seed, idx) for idx in range(self.k)]
        e1 = [self.sampler.sample_cbd(r_seed, self.k + idx) for idx in range(self.k)]
        e2 = self.sampler.sample_cbd(r_seed, 2 * self.k)

        ATr = self.vmm.mat_vec_mul(AT, r)
        u = self.vmm.vec_add(ATr, e1)

        tTr = self.vmm.vec_dot(t, r)
        m_bar = self.serializer.encode_message_to_poly(message)
        v = self.alu.poly_add(self.alu.poly_add(tTr, e2), m_bar)
        return (u, v)

    def decrypt(self, secret_key, ciphertext) -> bytes:
        u, v = ciphertext
        sTu = self.vmm.vec_dot(secret_key, u)
        w = self.alu.poly_sub(v, sTu)
        return self.serializer.decode_poly_to_message(w)
