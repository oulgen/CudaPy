import ctypes


class CudaTypeError (Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


class Type:
  def hstype(self):
    return self._hstype

  def ctype(self):
    return self._ctype


class Pointer(Type):
  def __init__ (self, type):
    self._hstype = "*" + type._hstype
    self._ctype = ctypes.c_void_p
    self._elemType = type

  def elemType(self):
    return self._elemType


class Void(Type):
  _hstype = "void"
  _ctype = ctypes.c_int


class Bool(Type):
  _hstype = "bool"
  _ctype = ctypes.c_bool


class Int(Type):
  _hstype = "int"
  _ctype = ctypes.c_int


class Float(Type):
  _hstype = "float"
  _ctype = ctypes.c_float


class Double(Type):
  _hstype = "double"
  _ctype = ctypes.c_double


def parseType(s):
  if s == "void":
    return Void
  if s == "bool":
    return Bool
  if s == "int":
    return Int
  if s == "float":
    return Float
  if s == "double":
    return Double

  if len(s) > 0 and s[-1] == '*':
    return Pointer(parseType(s[:-1]))

  raise CudaTypeError("invalid type: " + s)


def elemType(l):
  if hasattr(l, 'elemType'):
    return l.elemType()
  if len(l) < 1:
    return Int
  if type(l[0]) is int:
    return Int
  elif type(l[0]) is float:
    return Float


class dim3(ctypes.Structure):
  _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]

  def __init__(self, x, y = 1, z = 1):
    super(dim3, self).__init__(x, y, z)
