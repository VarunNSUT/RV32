import hashlib
from .Profiler import Profiler

class SymmetricCryptoEngine:
    """Maps directly to a hardware SHA3/Keccak IP core block."""
    
    @staticmethod
    @Profiler.profile("Keccak PRF shake 256")
    def prf_shake256(seed: bytes, nonce: int, out_length: int) -> bytes:
        """Hardware SHAKE-256 state machine generator."""
        h = hashlib.shake_256()
        h.update(seed + bytes([nonce]))
        return h.digest(out_length)

    @staticmethod
    @Profiler.profile("Keccak XOF shake 128")
    def xof_shake128(seed: bytes, i: int, j: int, out_length: int) -> bytes:
        """Hardware SHAKE-128 state machine generator for Matrix A expansion."""
        h = hashlib.shake_128()
        h.update(seed + bytes([i, j]))
        return h.digest(out_length)