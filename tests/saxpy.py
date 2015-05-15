import cudapy as cp

# saxpy : void (float alpha, float* X, float* Y)
def saxpy(alpha, X, Y):
  if idx < len(X):
    Y[idx] = alpha * X[idx] + Y[idx]

# Compile the kernel function
saxpyCall = cp.compile(saxpy)

X = map(float, range(100))
Y = map(float, range(100))

# Transfer Y to device memory
Y = cp.CudaArray(Y)
# Make the SAXPY call
saxpyCall(len(X))(5.0, X, Y)
# Convert the result back to Python list
result = Y.toList()

print result