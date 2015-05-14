from ctypes import *
from pkg_resources import resource_filename

import cudatypes


# Load CUDA memory library
libCudaPy = cdll.LoadLibrary(resource_filename(__name__, 'libCudaPy.so'))

cudaPyHostToDevice = getattr(libCudaPy, "cudaPyHostToDevice")
cudaPyHostToDevice.argtypes = [c_void_p, c_void_p, c_size_t, c_size_t]
cudaPyHostToDevice.restype = c_int;

cudaPyDeviceToHost = getattr(libCudaPy, "cudaPyDeviceToHost")
cudaPyDeviceToHost.argtypes = [c_void_p, c_void_p, c_size_t, c_size_t]
cudaPyDeviceToHost.restype = c_int;

cudaPyAllocArray = getattr(libCudaPy, "cudaPyAllocArray")
cudaPyAllocArray.argtypes = [c_size_t, c_size_t]
cudaPyAllocArray.restype = c_void_p;

cudaPyFree = getattr(libCudaPy, "cudaPyFree")
cudaPyFree.argtypes = [c_void_p]
cudaPyFree.restype = c_int;


class CudaError (Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


class BaseCudaArray(object):
  def __init__(self, size, type):
    self._length = size
    self._cudaType = type
    self._type = type._ctype
    self._pointer = cudaPyAllocArray(self._length, sizeof(self._type))
    if self._pointer is None:
      raise CudaError("bad CUDA malloc")

  def toHost(self):
    out = (self._type * self._length)()
    code = cudaPyDeviceToHost(cast(out, c_void_p), self._pointer, self._length, sizeof(self._type))
    if code != 0:
      raise CudaError("failed to copy from device to host: " + str(self._pointer))
    return out

  def toList(self):
    out = self.toHost()
    return [out[i] for i in xrange(self._length)]

  def __len__(self):
    return self._length

  def length(self):
    return self._length

  def elemType(self):
    return self._cudaType

  def pointer(self):
    return self._pointer

  def __getitem__(self, i):
    pass

  def __setitem__(self, i, v):
    pass

  def __del__(self):
    cudaPyFree(self._pointer)


class CudaArray(BaseCudaArray):
  def __init__(self, l):
    super(CudaArray, self).__init__(len(l), cudatypes.elemType(l))
    tmp = (self._type * len(l))(*l)
    code = cudaPyHostToDevice(self._pointer, tmp, len(l), sizeof(self._type))
    if code != 0:
      raise CudaError("failed to copy from host to device: " + str(self._pointer))

  @staticmethod
  def allocate(size, type):
    return BaseCudaArray(size, type)
