# CudaPy module
from ctypes import *
import hashlib, subprocess, tempfile, os.path
from pkg_resources import resource_filename

from cudatypes import *
from template import Template, parseSig
from wrapper import wrapper


# Load the py2cuda library
py2cudaLib = cdll.LoadLibrary(resource_filename(__name__, 'py2cuda.so'))
py2cudaExtern = getattr(py2cudaLib, "py2cuda")
py2cudaExtern.argtypes = [c_char_p, c_char_p]
py2cudaExtern.restype = c_char_p


class CudaPyError (Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


def kernel(sig = None, debug = False):
  def inner(f):
    return compile(f, sig, debug)

  return inner


def compile(funs, sigs = None, debug = False):
  if not isinstance(funs, list):
    return compile([funs], [sigs], debug)

  if len(funs) == 0:
    return None

  if sigs is None:
    sigs = [None] * len(funs)

  (pySources, sigs) = zip(*map(getSource, funs, sigs))
  pySource = "\n\n".join(pySources)

  (funExt, sigExt) = zip(*[(fun, sig) for (fun, sig) in zip(funs, sigs) if sig[0] is Void])
  funNames = [fun.__name__ for fun in funExt]
  debugOut = funNames[0] if len(funNames) > 0 else 'module'
  cudaSource = py2cuda(pySource, sigs, output = debugOut + ".cu" if debug else None)
  cudaCalls = compileCuda(cudaSource, sigExt, ["__call" + f for f in funNames])

  return wrapper(cudaCalls[0], sigExt[0], funNames[0])


# Returns the source code and type signature of the given function object
def getSource(fun, sig = None):
  if not isinstance(fun, Template):
    fun = Template(fun)

  if isinstance(sig, list):
    pass
  elif isinstance(sig, basestring):
    sig = parseSig(fun.__name__ + " : " + sig, fun.__name__)
  else:
    sig = fun._signature

  if sig is None:
    raise CudaPyError("function does not have a valid signature: " + fun.__name__)

  return (fun._source, sig)


def py2cuda(source, sigs, output = None):
  hstypes = [[t._hstype for t in sig] for sig in sigs]
  sigEnc = '\n'.join([' '.join(sig) for sig in hstypes])
  cudaSource = py2cudaExtern(source, sigEnc)

  # Check for errors during translation
  [code, cudaSource] = cudaSource.split(':', 1)
  if code == "error":
    raise CudaPyError(cudaSource)

  if output is not None:
    with open(output, "w") as f:
      f.write(cudaSource)

  return cudaSource


def compileCuda(source, sigs, funNames):
  libFile = hash(source) + ".so"

  if not os.path.isfile(libFile):
    flags = ["-O3"]
    shared = ["--shared", "--compiler-options", "-fPIC", "-x", "cu"]
    warnings = [ "-Xcudafe"
               , "--diag_suppress=declared_but_not_referenced"
               , "-Xcudafe"
               , "--diag_suppress=set_but_not_used"
               ]

    tmpFile = tempfile.NamedTemporaryFile(suffix = '.cu')

    tmpFile.write(source)
    tmpFile.seek(0)

    try:
      files = ["-o", libFile, tmpFile.name]
      subprocess.check_output(["nvcc"] + flags + shared + warnings + files)
    except subprocess.CalledProcessError as e:
      print e.output
      raise CudaPyError("nvcc exited with error code " + str(e.returncode))
    finally:
      tmpFile.close()

  funs = []
  for (sig, funName) in zip(sigs, funNames):
    fun = getattr(cdll.LoadLibrary(libFile), funName)
    fun.restype = sig[0]._ctype
    fun.argtypes = [dim3, dim3] + [t._ctype for t in sig[1:]]
    funs.append(fun)

  return funs


def hash(str):
  return hashlib.sha224(str).hexdigest()[:32]
