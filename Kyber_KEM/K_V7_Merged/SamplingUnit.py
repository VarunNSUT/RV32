from .SymmetricCrypto import SymmetricCryptoEngine
from .ArithmeticUnit import Poly
import hashlib

class SamplingUnit:
    """Maps to filtering logic that parses byte streams into valid ring coefficients."""
    
    def __init__(self, n: int, q: int, eta: int):
        self.N = n
        self.Q = q
        self.ETA = eta

    def sample_uniform(self, seed: bytes, i: int, j: int) -> Poly:
        """Rejection sampling logic. Implements variable-latency hardware loop."""
        # Over-generate bytes to handle hardware streaming pipeline
        stream = SymmetricCryptoEngine.xof_shake128(seed, i, j, 3 * self.N * 2 + 32)
        result = []
        idx = 0
        while len(result) < self.N:
            if idx + 2 >= len(stream):
                # Refill buffer constraint simulator
                h2 = hashlib.shake_128()
                h2.update(stream[-32:])
                stream += h2.digest(3 * self.N)
            b0, b1, b2 = stream[idx], stream[idx + 1], stream[idx + 2]
            idx += 3
            d1 = b0 + 256 * (b1 & 0x0F)
            d2 = (b1 >> 4) + 16 * b2
            if d1 < self.Q:
                result.append(d1)
            if d2 < self.Q and len(result) < self.N:
                result.append(d2)
        return result[:self.N]

    def sample_cbd(self, seed: bytes, nonce: int) -> Poly:
        """Centered Binomial Distribution. Maps to bit-counting adder trees."""
        num_bytes = ((2 * self.ETA * self.N) + 7) // 8
        buf = SymmetricCryptoEngine.prf_shake256(seed, nonce, num_bytes)
        bits = []
        for byte in buf:
            for bit_pos in range(8):
                bits.append((byte >> bit_pos) & 1)
        coeffs = []
        for i in range(self.N):
            a = sum(bits[2 * self.ETA * i + j] for j in range(self.ETA))
            b = sum(bits[2 * self.ETA * i + self.ETA + j] for j in range(self.ETA))
            coeffs.append((a - b) % self.Q)
        return coeffs