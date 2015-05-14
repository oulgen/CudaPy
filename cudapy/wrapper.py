from cudatypes import dim3, Pointer
from cudaarray import BaseCudaArray, CudaArray


class CudaPyError (Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


# Handle grid and block dims, coerce CudaArrays into bare pointers
def wrapper(fun, sig, funName):
  threadSize = 512
  sig = sig[1:]

  def kernel(callDim, y = 1, z = 1):
    if not isinstance(callDim, dim3):
      callDim = dim3(callDim, y, z)
    blockDim = allocateThreads(threadSize, callDim)
    gridDim = getGridDim(callDim, blockDim)

    def coerceArgs(*args):
      args = list(args)
      if len(args) != len(sig):
        raise CudaPyError(funName + " takes " + str(len(sig)) + " arguments.")

      temps = [] # Prevent premature garbage collection
      for i in xrange(len(sig)):
        if isinstance(sig[i], Pointer):
          if isinstance(args[i], list):
            temps.append(CudaArray(args[i]))
            args[i] = temps[-1]
          assert isinstance(args[i], BaseCudaArray), "expected CudaArray found " + type(args[i]).__name__
          assert args[i].elemType() == sig[i].elemType(), "argument types do not match"
          args[i] = args[i].pointer()

      args = [gridDim, blockDim] + args
      fun(*args)

    return coerceArgs

  return kernel;


# Allocate available threads to three dimensions
def allocateThreads(threads, dim):
  def power_two(n):
    return 1 << (n.bit_length() - 1)

  tx = min(threads, power_two(dim.x))
  threads //= tx
  ty = min(threads, power_two(dim.y))
  threads //= ty
  tz = min(threads, power_two(dim.z))
  threads //= tz

  return dim3(tx, ty, tz)


# Compute grid dimensions from data and block dimensions
def getGridDim(callDim, blockDim):
  def divideUp(n, d):
    return (n + d - 1) // d

  x = divideUp(callDim.x, blockDim.x)
  y = divideUp(callDim.y, blockDim.y)
  z = divideUp(callDim.z, blockDim.z)
  return dim3(x, y, z)
