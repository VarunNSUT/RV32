from typing import List
from .ArithmeticUnit import ArithmeticUnit, ntt, intt, ntt_mul
from .SamplingUnit import SamplingUnit

Poly = List[int]
Vec  = List[Poly]
Mat  = List[Vec]


class VectorMatrixMultiplier:
    """
    Orchestrates multi-polynomial pipelines across multi-ported memory blocks.
    A matrix is stored in NTT domain — mat_vec_mul skips redundant fwd NTTs.
    """

    def __init__(self, alu: ArithmeticUnit):
        self.alu = alu

    def vec_add(self, a: Vec, b: Vec) -> Vec:
        return [self.alu.poly_add(a[i], b[i]) for i in range(len(a))]

    def vec_dot(self, a: Vec, b: Vec,
                a_in_ntt: bool = False, b_in_ntt: bool = False) -> Poly:
        acc = self.alu.poly_zero()
        for ai, bi in zip(a, b):
            a_ntt = ai if a_in_ntt else ntt(ai)
            b_ntt = bi if b_in_ntt else ntt(bi)
            acc = self.alu.poly_add(acc, intt(ntt_mul(a_ntt, b_ntt)))
        return acc

    def mat_vec_mul(self, A: Mat, v: Vec,
                    A_in_ntt: bool = False, v_in_ntt: bool = False) -> Vec:
        """
        Matrix-vector multiply.
        A_in_ntt / v_in_ntt flags skip redundant forward NTTs when the caller
        has already transformed the inputs — mirrors V5 NTT-domain caching.
        """
        v_ntt = v if v_in_ntt else [ntt(vi) for vi in v]
        result = []
        for row in A:
            acc = self.alu.poly_zero()
            for aij, vj_ntt in zip(row, v_ntt):
                a_ntt = aij if A_in_ntt else ntt(aij)
                acc = self.alu.poly_add(acc, intt(ntt_mul(a_ntt, vj_ntt)))
            result.append(acc)
        return result

    def mat_transpose(self, A: Mat) -> Mat:
        k = len(A)
        return [[A[j][i] for j in range(k)] for i in range(k)]

    def generate_A_matrix(self, sampler: SamplingUnit,
                          rho: bytes, k: int,
                          store_ntt: bool = True) -> Mat:
        """
        Generate public matrix A from rho.
        store_ntt=True  → each polynomial stored directly in NTT domain,
                          so mat_vec_mul can pass A_in_ntt=True and skip
                          k² forward NTTs on every encrypt/decrypt call.
        """
        A = []
        for i in range(k):
            row = []
            for j in range(k):
                poly = sampler.sample_uniform(rho, i, j)
                row.append(ntt(poly) if store_ntt else poly)
            A.append(row)
        return A
