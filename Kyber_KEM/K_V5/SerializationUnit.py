from .ArithmeticUnit import Poly

class SerializationUnit:
    """Handles data transformation between physical bit busses and polynomial coefficients."""
    
    def __init__(self, n: int, q: int):
        self.N = n
        self.Q = q
        self.chunk_size = (n + 7) // 8

    def encode_message_to_poly(self, m: bytes) -> Poly:
        assert len(m) == self.chunk_size, f"Hardware expect exact byte payload interface alignment."
        half_q = self.Q // 2
        coeffs = []
        for byte in m:
            for bit in range(8):
                if len(coeffs) < self.N:
                    coeffs.append(half_q if (byte >> bit) & 1 else 0)
        return coeffs

    def decode_poly_to_message(self, w: Poly) -> bytes:
        half_q = self.Q // 2
        bits = []
        for coef in w:
            c = coef % self.Q
            dist0 = min(c, self.Q - c)
            dist_half = abs(c - half_q)
            bits.append(1 if dist_half < dist0 else 0)
            
        result = bytearray(self.chunk_size)
        for i, bit in enumerate(bits):
            result[i // 8] |= bit << (i % 8)
        return bytes(result)
