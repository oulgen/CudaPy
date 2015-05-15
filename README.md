# Summary
CudaPy is a runtime library that lets Python programmers access NVIDIA's CUDA parallel computation API. It lets you write CUDA kernels in Python, and provides a nice API to invoke them. It works by translating CUDA kernels written in Python to C++, and JIT compiling them using `nvcc`. CudaPy offers many conveniences compared to C++ CUDA, and has many advantages over similar wrapper libraries for Python:

* Native: You do not have to write or even see C++ code. CudaPy kernels are written purely in (a [subset](#kernel-functions) of) Python. Kernel invocation is done by calling Python functions.
* Dynamic: No compilation is required. Just start your interpreter and CudaPy will JIT compile your kernels.
* Efficient: CudaPy JIT compiles kernels to native code, so kernel calls are as efficient as C++ CUDA. Copying of data to and from the device memory is done only when necessary, and can be controlled manually.
* Convenient: CudaPy handles many hurdles that other CUDA programmers have to deal with manually. These include automatic allocation of threads per block ([Kernel Invocation](#kernel-invocation)), management of device memory lifetimes ([Cuda Arrays](#cuda-arrays)), and transferring of data to and from the device.
* Safe: CudaPy kernels are typed. Kernels are statically type-checked based on function signatures that you provide. All invocations of kernels dynamically verify input types.
* Extensible: CudaPy features a (basic) template system that allows the definition of higher-order functions like `map`, `zipWith`, `scan` etc. (Note that this feature is still experimental).


# Background

Python is **slow**. Python programmers who need efficiency generally resort to using libraries like `numpy`, which are simply wrappers around compiled C code. Add thread level parallelism to that and maybe you start using the CPU efficiently, but most machines offer more compute capability in the form of a GPU. This is where CudaPy comes in. CudaPy introduces GPU parallelism to Python by providing an efficient, native, and easy to use interface to CUDA.

Our cursory research on CUDA APIs for Python left much to be desired. Some libraries only allowed access to predefined functions sacrificing extensibility. Others were not "Pythony" enough: one library we found actually used quoted C code ([PyCuda](http://mathema.tician.de/software/pycuda/)). Other libraries required the use of a compiler, which went against the dynamic nature of Python ([NumbaPro](http://docs.continuum.io/numbapro/)).

We believe in simplicity and usability. For a concrete example, this is how we would like to implement SAXPY in Python:
```
import cudapy as cp

# saxpy : void (float alpha, float* X, float* Y)
def saxpy(alpha, X, Y):
  if idx < len(X):
    Y[idx] = alpha * X[idx] + Y[idx]

# Compile the kernel function
saxpyCall = cp.compile(saxpy)

alpha = ... # Pick a float
X = ...     # Create and populate Python list X
Y = ...     # Create and populate Python list Y

# Transfer Y to device memory
Y = cp.CudaArray(Y)
# Make the SAXPY call
saxpyCall(len(X))(alpha, X, Y)
# Convert the result back to Python list
result = Y.toList()
```
You should note a few things here. First of all, this is pure Python. Second, the kernel call (`saxpyCall`) does not take in grid or block dimensions; it only takes in the number of threads we want. Finally, even though we had to copy `Y` to device memory manually to be able to refer to it later, `X` is copied automatically. CudaPy handles most memory operations automatically, and provides finer control when we need it.

At its heart, CudaPy does the following: it takes a kernel function (and possibly a list of helper functions) written in Python and its type signature, and returns a Python function that invokes this kernel. Apart from that, CudaPy provides nice conveniences like automatic thread allocation and a class to manage memory.

# Approach

Here is an overview of the CudaPy "production line":

1. `cudapy.compile` is given a kernel function and possibly helper functions.
2. For each function, retrieve its source code using Python's inspection facilities. Parse the function's type signature from comments above it.
3. The source and type signatures are sent to a shared library (written in Haskell) using Python's foreign function interface.
4. Parse the raw source into an AST using `language-python`, an external Haskell package.
5. Translate Python AST to a C like AST with type information.
6. Infer types for variable declarations and type-check functions.
7. Put together the CUDA source. This involves rendering our AST as C source code, adding `__global__` and `__device__` declarations, forward declaring all functions (all functions in CudaPy are mutually recursive), and creating kernel invocation functions.
8. Python receives the CUDA source (by the foreign function interface), and compiles it using `nvcc` into a shared library.
9. Dynamically load the kernel invocation function using `ctypes`.
10. Wrap a convenience function around this (to handle device memory and thread allocation) and return it to the user.

CudaPy caches compiled functions, so step 8 (which is by far the slowest step) is skipped after the first time a code is ran.

Here is an example translation. First, recall the SAXPY kernel from before:
```
# saxpy : void (float alpha, float* X, float* Y)
def saxpy(alpha, X, Y):
  if idx < len(X):
    Y[idx] = alpha * X[idx] + Y[idx]
```

Given this, our library produces the following CUDA C++ code:
```
__global__ void saxpy (int, int*, int*, int*);

extern "C" {
  void __callsaxpy (dim3, dim3, int, int*, int*, int*);
}

__device__ static
inline size_t len(void* arr)
{
  return *((size_t*)arr - 1);
}

__global__
void saxpy (int alpha, int* X, int* Y, int* result)
{
  int idx;
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len(X))
  {
    result[idx] = alpha * X[idx] + Y[idx];
  }
}

void __callsaxpy (dim3 gridDim, dim3 blockDim, int arg0, int* arg1,
                  int* arg2, int* arg3)
{
  saxpy<<<gridDim, blockDim>>>(arg0, arg1, arg2, arg3);
  cudaThreadSynchronize();
}
```

As you can see, the kernel is pretty much exactly mirrored as the `saxpy` function. Our library infers that it needs to be global. We also create the `__callsaxpy` function, which invokes the kernel and immediately synchronizes. The first few lines forward declare all functions, export `__callxxxx` functions to C, and define our library functions (only `len` in this case).

## Kernel Functions

A kernel function is simply a Python function (with some restrictions), that looks like this:
```
# saxpy : void (float alpha, float* X, float* Y)
def saxpy(alpha, X, Y):
  if idx < len(X):
    Y[idx] = alpha * X[idx] + Y[idx]
```
The only thing not Python about kernel functions is some predefined CUDA constants and a type signature.

The type signature is provided in a comment above each kernel function. It can also be overwritten by supplying a `sig` argument to `cudapy.compile`. Valid types are the following: `void`, `bool`, `int`, `float`, `double`, `t*` where `t` is a type. Kernels must have a `void` return type. Helper functions can have any type.

As for the kernel function itself, we support the following subset of Python/CUDA:

* Predefined constants: `gridDim` (`gridDim.x`, ...), `blockDim`, `blockIdx`, `threadIdx`, `warpSize`, `idx`, `idy`, `idz`. The first five constants have the same meaning as in CUDA. `idx` etc. are just aliases for `blockIdx.x * blockDim.x + threadIdx.x`. All these have the type `int`.
* Constants: Integer (e.g `42`), floating point (`4.2`), boolean (`True`, `False`)
* Operators: `+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `&`, `|`, `^`, unary `-` (negation), `~`,  `==`, `!=`, `<`, `<=`, `>`, `>=`, `and`, `or`, `not`.
* Conditional statements: `x if b else y`
* Array operations: `len` that returns the length of an array. The length of an array cannot be changed.
* Mathematical functions: Single and double precision library functions supported by CUDA. Examples: `math.sin`, `math.sinf`...
* Control flow: while loops, if statements, `continue`, `break`, C like for loops (`for i in xrange(start, end, step)`) etc.
* Casts: `int`, `float`, `double`

Since Python does not have scoping over variables, each variable has to be used exactly at one type.

## Compiling Kernels
CUDA kernel functions can be compiled very easily by calling `cp.compile([function1, function2, ...])`. For example, the previous `saxpy` function can be compiled by
```
saxpyCall = cp.compile(saxpy)
```

If your kernel call uses other device functions as helper functions, they need to be compiled together. For example, if our SAXPY kernel function used a helper function called `add`, we would compile it in the following way:
```
saxpyWithHelperCall = cp.compile([saxpy, add])
```

## Kernel Invocation
Kernel invocation is similarly easy. Remember how we invoked the `saxpy` kernel:
```
saxpyCall(len(X))(alpha, X, Y)
```

`cudaarray.compile` takes a list of functions and returns a curried function. The first argument is the dimensions: how many threads you want for x, y, and z. This will usually be the dimensions of your data. For example, we pass the length of `X` to `saxpyCall`. If you provide less than 3 dimensions, the default for `y` and `z` is 1.


## CUDA Arrays
CudaPy provides a CudaArray class that handles the transfer of data between the host and the device. The CudaArray class provides a nice and clear interface to the device memory and hides ugly library calls like `cudaMalloc` and `cudaMemcpy`. It also handles garbage collection through object lifetimes, so the user does not have to worry about freeing device memory.

There are two ways to create an array that resides in device memory:

* Allocate an empty array: `cudapy.CudaArray.allocate(length, type)`
* Copy a Python list: `cudapy.CudaArray(list)`

CudaArrays are transfered back to host memory by using the `.toList()` method.

Once created, CudaArrays can be passed to compiled kernels as arguments and they will persist through kernel invocations. In general, you do not need to create CudaArrays manually for inputs as kernel calls automatically transfer Python lists to device memory. You only need to use CudaArrays when you need to refer to them (for example, in function results), or to avoid copying the same array to the device multiple times.

## Memory Management
CudaPy provides finer control over data when you need it. CudaArrays do not copy device memory back unless you invoke the `.toList()` method. This means you can chain kernel calls on the same CudaArray without moving data back and forth between the host and the device memory. Take the following example:

```
X = # Create and populate X
Y = # Create and populate Y

# Create the result array
cudaResult = cp.CudaArray.allocate(len(X), cp.Int)

# Make a multiply call
multiplyCall(len(X))(alpha, X, cudaResult)

# Make an add call
addCall(len(X))(Y, cudaResult)

# Convert the result back to Python list
result = cudaResult.toList()
```

Assuming `multiplyCall` and `addCall` are defined, this is a perfectly fine way of implementing SAXPY in terms of correctness. Note that `cudaResult` lives in device memory until the last line, and it is copied back only once. `X` is copied from host to device during `multiplyCall` and `Y` is copied during `addCall`. Assuming they go out of scope, they are garbage collected a never copied back.

## Limitations & Future Work
CudaPy has a couple limitations:

* Python limitations: Nested functions and objects are not supported. Nested functions would degrade performance, and require arbitrary jumps and garbage collection on the GPU. Objects (or a subset of objects) could be supported, but would make typing much more complicated. A nice alternative could be C-like structs, which should be east to implement.
* Built-in functions: CudaPy is basically a translator, thus it needs access to the source code of a function. Python cannot retrieve the source code of built-in functions, so they cannot be used with CudaPy. This also means CudaPy cannot compile a function if it cannot access its source (there could be multiple reasons for this). This should not be a problem for most programs since kernel functions and calls to `cudapy.compile` generally reside in the same file.
* Shared memory: CudaPy does not support shared memory. This wasn't a design decision or particularly hard, we simply decided to concentrate on other features. We have the basic idea and a future release could incorporate it. (Shared memory is a big part of CUDA, so this would be a priority).

# Results
In this section, we want to give a general idea of just how fast CudaPy is. This means comparing execution times, and there are many ways to do that. We could compare CudaPy with vanilla Python, but this is not fair since CudaPy is compiled. Handwritten CUDA vs CudaPy is not instructive, as these will generate pretty much the same code. Since we are trying to make Python faster, it makes sense to compare CudaPy to current high-speed libraries. For this reason, we choose to compare CudaPy with NumPy.

We implemented several functions using CudaPy and NumPy in what we think is the most efficient way. We then timed the execution of these programs over large datasets for many iterations. Below, we give the total execution time of each function over the same dataset. The running time of matrix multiplication and Mandelbrot include the cost of copying data to and from the device memory. Since SAXPY is bandwidth bounded, this would make no sense so copying costs are excluded. This is justified since SAXPY could be part of a larger computation, and the intermediate date would be kept on device memory.

![Timing](img/runtime.png)
![Speedup](img/speedup.png)


These results were attained on the following computer:
```
Macbook Pro Retina (Mid 2012)
OS X Yosemite (10.10.3)
2.3 GHz Intel Core i7
8 GB 1600 MHz DDR3

NVIDIA GeForce GT 650M
CUDA cores: 384 cores
Graphics Clock (MHz): Up to 900 MHz
Memory Bandwidth (GB/sec): Up to 80.0
Bus: PCIe
Bus width: x8
VRAM: 1024MB
```

Here is the Mandelbrot image:
![Mandel](img/mandel.png)


# Related Work

We got our inspiration for CudaPy form a system called VecPy. VecPy was last year's winner at [15-418 Parallelism competition](http://15418.courses.cs.cmu.edu/spring2014/competition). As in its creator's words, VecPy "leverages multi-threading and SIMD instructions on modern x86 processors." CudaPy goes in a different direction and adds GPU level parallelism. We also have a less strict type system: VecPy compiles code for a single type like `int` or `float` where as CudaPy kernels can take an arbitrary signature of base types (these include `void`, `bool`, `int`, `float`, `double`, and possibly nested arrays on these types). Finally, CudaPy had some extra challenges VecPy did not have such as handling separate device and host memory, and interfacing with the CUDA runtime.

# References