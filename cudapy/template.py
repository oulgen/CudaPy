import inspect, re
from textwrap import dedent
from types import FunctionType

from cudatypes import parseType


class TemplateError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


class Template:
  def __init__(self, fun):
    # Make sure we are given a function and try to retrieve its source code
    if not inspect.isfunction(fun):
      raise TemplateError("object not a function: " + str(fun))
    try:
      source = stripDecorator(dedent(inspect.getsource(fun)))
    except:
      raise TemplateError("cannot retrieve the source code of " + str(fun))

    # Get type signature
    try:
      candidates = inspect.getcomments(fun).splitlines()
      candidatesP = [parseSig(s, fun.__name__) for s in candidates]
      sig = next((s for s in candidatesP if s is not None), None)
    except:
      sig = None

    # Store template
    self.__name__ = fun.__name__
    self._function = fun
    self._source = source
    self._signature = sig

  def substitute(self, subst):
    if isinstance(subst, list):
      for sub in subst:
        self.substitute(sub)
      return
    self.__replace(subst[1], self.__repr(subst[0]))
    return self

  # String replacement for whole worlds
  def __replace(self, old, new):
    self._source = re.sub(r'\b' + old + r'\b', new, self._source)
    return self

  @staticmethod
  def __repr(obj):
    if isinstance(obj, FunctionType):
      return obj.__name__
    return str(obj)


def template(subs):
  def inner(f):
    return Template(f).substitute(subs)

  return inner


# Returns None if not a valid signature
def parseSig(sig, funName):
  pat = r"#? \s* (\w+) \s*:\s* (\w+) \s* \( (.*) \)".replace(" ", "")
  m = re.match(pat, sig)
  if not m or m.group(1) != funName:
    return None

  restype = m.group(2)
  args = [x.strip() for x in m.group(3).split(',')]
  argtypes = [arg.split(' ')[0] for arg in args]

  try:
    restype = parseType(restype)
    argtypes = [parseType(t) for t in argtypes]
  except:
    return None

  return [restype] + argtypes


# Strip decorators from function source
def stripDecorator(source):
  # Optimize common case
  if source.startswith("def"):
    return source

  lines = source.splitlines()
  keep = []
  for i in xrange(len(lines)):
    if lines[i].lstrip().startswith("def"):
      break
    if lines[i].lstrip().startswith("#"):
      keep.append(lines[i])
  return '\n'.join(keep + lines[i:])
