from typing import List
Poly = List[int]

class ArithmeticUnit:
    """Maps directly to the physical DSP slices and combinational logic paths."""
    
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

    def poly_mul_schoolbook(self, a: Poly, b: Poly) -> Poly:
        """
        Unoptimized Ring Multiplication loop.
        Warning: In hardware, this maps to N^2 physical multipliers if unrolled.
        """
        result = [0] * self.N
        for i in range(self.N):
            if a[i] == 0: continue
            for j in range(self.N):
                if b[j] == 0: continue
                deg = i + j
                coef = a[i] * b[j]
                if deg < self.N:
                    result[deg] = (result[deg] + coef) % self.Q
                else:
                    # Modular reduction step for the ring X^N + 1 (sign flip subtraction)
                    result[deg - self.N] = (result[deg - self.N] - coef) % self.Q
        return result

    def centered_mod(self, a: int) -> int:
        r = a % self.Q
        if r > self.Q // 2:
            r -= self.Q
        return r
