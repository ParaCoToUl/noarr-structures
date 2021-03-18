# Interesting kernels

### Něco, co neco skutečně dělá
``` cpp
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_calc_average_distance(
    uint32_t offset, uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ assignments,
    atomic_float *distance) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    sample += offset;
    dist = METRIC<M, F>::distance_t(
        samples, centroids + assignments[sample] * d_features_size,
        d_samples_size, sample);
  }
  float sum = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(distance, sum);
  }
}
```


### Common look of some kernels
``` cpp
// Common reduction kernel that aggregates all privatized copies into one.
template<typename F, typename IDX_T, class LAYOUT_MEANS>
__global__ void divideMeansKernel(F* __restrict__ means, const IDX_T* __restrict__ clusterSizes, IDX_T dim, IDX_T k)
{
	auto size = k * dim;
	auto preK = LAYOUT_MEANS::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	for (IDX_T i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		IDX_T d = i % dim;
		IDX_T idx = i / dim;
		auto divisor = clusterSizes[idx];
		if (divisor > 0) {
			LAYOUT_MEANS::at(means, idx, d, preK) /= (F)divisor;
		}
	}
}
```

### transpose()
``` cpp
template <bool xyswap>
__global__ void transpose(
    const float *__restrict__ input, uint32_t rows, uint32_t cols,
    float *__restrict__ output) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  volatile uint32_t x = xyswap?
      blockIdx.y * TILE_DIM + threadIdx.y:
      blockIdx.x * TILE_DIM + threadIdx.x;
  volatile uint32_t y = xyswap?
      blockIdx.x * TILE_DIM + threadIdx.x:
      blockIdx.y * TILE_DIM + threadIdx.y;
  volatile uint32_t tx = xyswap? threadIdx.y : threadIdx.x;
  volatile uint32_t ty = xyswap? threadIdx.x : threadIdx.y;

  if (x < cols && y < rows) {
    for (uint32_t j = 0;
         j < min(static_cast<unsigned int>(TILE_DIM), rows - y);
         j += BLOCK_ROWS) {
      tile[ty + j][tx] = input[static_cast<uint64_t>(y + j) * cols + x];
    }
  }

  __syncthreads();

  x = xyswap?
      blockIdx.x * TILE_DIM + threadIdx.y:
      blockIdx.y * TILE_DIM + threadIdx.x;
  y = xyswap?
      blockIdx.y * TILE_DIM + threadIdx.x:
      blockIdx.x * TILE_DIM + threadIdx.y;

  if (x < rows && y < cols) {
    for (uint32_t j = 0;
         j < min(static_cast<unsigned int>(TILE_DIM), cols - y);
         j += BLOCK_ROWS) {
      output[static_cast<uint64_t>(y + j) * rows + x] = tile[tx][ty + j];
    }
  }
}
```
