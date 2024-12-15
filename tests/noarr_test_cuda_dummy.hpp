#ifndef NOARR_TEST_CUDA_DUMMY_HPP
#define NOARR_TEST_CUDA_DUMMY_HPP


#define __device__

typedef unsigned uint;

[[maybe_unused]] static struct dim3 { uint x = 1, y = 1, z = 1; } threadIdx, blockIdx, blockDim;


#endif // NOARR_TEST_CUDA_DUMMY_HPP
