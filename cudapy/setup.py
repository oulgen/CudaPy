
from setuptools import setup, find_packages

setup(
  name = "CudaPy",
  version = "0.1",
  packages = find_packages(),
  package_data = {
    '': ['*.so'],
  },
  zip_safe = True,

  author = "Josh Acay, Oguz Ulgen",
  author_email = "cacay@cmu.edu",
  license = "MIT",
  url = "http://418.oulgen.com/",
)