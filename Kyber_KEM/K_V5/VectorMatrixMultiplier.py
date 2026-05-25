from .ArithmeticUnit import ArithmeticUnit, Poly
from .SamplingUnit import SamplingUnit
from typing import List

Vec = List[Poly]
Mat = List[Vec]

class VectorMatrixMultiplier:
    """Orchestrates multi-polynomial pipelines across multi-ported memory blocks."""
    
    def __init__(self, alu: ArithmeticUnit):
        self.alu = alu

    def vec_add(self, a: Vec, b: Vec) -> Vec:
        return [self.alu.poly_add(a[i], b[i]) for i in range(len(a))]

    def vec_dot(self, a: Vec, b: Vec) -> Poly:
        acc = self.alu.poly_zero()
        for ai, bi in zip(a, b):
            acc = self.alu.poly_add(acc, self.alu.poly_mul_schoolbook(ai, bi))
        return acc

    def mat_vec_mul(self, A: Mat, v: Vec) -> Vec:
        k = len(A)
        return [self.vec_dot(A[i], v) for i in range(k)]

    def mat_transpose(self, A: Mat) -> Mat:
        k = len(A)
        return [[A[j][i] for j in range(k)] for i in range(k)]

    def generate_A_matrix(self, sampler: SamplingUnit, rho: bytes, k: int) -> Mat:
        return [[sampler.sample_uniform(rho, i, j) for j in range(k)] for i in range(k)]
