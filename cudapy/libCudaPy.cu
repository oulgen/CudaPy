#include <cuda.h>

extern "C" {
  int cudaPyHostToDevice(void*, void*, size_t, size_t);
  int cudaPyDeviceToHost(void*, void*, size_t, size_t);
  void* cudaPyAllocArray(size_t, size_t);
  int cudaPyFree(void*);
}


int cudaPyHostToDevice(void* dst, void* src, size_t N, size_t tsize) {
  return cudaMemcpy(dst, src, N * tsize, cudaMemcpyHostToDevice);
}


int cudaPyDeviceToHost(void* dst, void* src, size_t N, size_t tsize) {
  return cudaMemcpy(dst, src, N * tsize, cudaMemcpyDeviceToHost);
}


void* cudaPyAllocArray(size_t N, size_t tsize) {
  void* arr;
  size_t arraySize = 2 * sizeof(size_t) + N * tsize;

  if (cudaMalloc(&arr, arraySize))
    return NULL;
  cudaMemset(&arr, 0, arraySize);

  size_t header[2] = {tsize, N};
  cudaMemcpy(arr, &header, sizeof(size_t) * sizeof(header), cudaMemcpyHostToDevice);

  return (void*)((size_t*)arr + 2);
}


int cudaPyFree(void* input) {
  return cudaFree((void*)((size_t*)input - 2));
}
