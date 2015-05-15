import cudapy as cp

from time import time

# rangeId : void (float* A)
def rangeId(A):
  if idx < len(A):
    A[idx] = float(idx)

__rangeIdCall = cp.compile(rangeId)

# matrixMultiply : void (float* A, float* B, float* C, int m, int n, int p)
# A is m x n
# B is n x p
# C is m x p
def matrixMultiply(A, B, C, m, n, p):
  row = idy
  col = idx
  result = 0.0

  if row >= m or col >= p:
    return

  for i in xrange(n):
    result += A[row * n + i] * B[i * p + col]

  C[row * p + col] = result

__matrixMultiplyCall = cp.compile(matrixMultiply)

m = 1200
n = 800
p = 1000

A = cp.CudaArray.allocate(m * n, cp.Float)
B = cp.CudaArray.allocate(n * p, cp.Float)
__rangeIdCall(m * n)(A)
__rangeIdCall(n * p)(B)

start = time()
C = cp.CudaArray.allocate(m * p, cp.Float)
__matrixMultiplyCall(p, m)(A, B, C, m, n, p)
C = C.toHost()
total = time() - start
print "Total time: ", total

print C[200]
