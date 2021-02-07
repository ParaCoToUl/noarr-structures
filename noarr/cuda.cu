#include <array>
#include <iostream>

#if _WIN32 || _WIN64
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define SIZE_CONSTANT 10000
#else
#define SIZE_CONSTANT 20000
#endif

#include "noarr_funcs.hpp"

using namespace noarr;

// same body, two data layouts:
template<typename AS>
__global__ void kernel(float *data, AS as) {
    auto index = as % fixs<'y', 'x'>(blockIdx.x, threadIdx.x);
    *(float*)((char*)data + (index % offset())) = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void kernel_handmade(float *data, size_t size) {
    data[blockIdx.x * size + threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
    std::array<float, 20 * SIZE_CONSTANT> local;
    float *data;

    cudaMalloc(&data, sizeof(local));

    const auto av = array<'y', SIZE_CONSTANT, vector<'x', scalar<float>>>{};
    volatile std::size_t s = 20;
    const auto avr = av % resize<'x'>(s);
    kernel<<<SIZE_CONSTANT, 20>>>(data, avr);
    //kernel_handmade<<<SIZE_CONSTANT, 20>>>(data, 20);
    
    cudaMemcpy(local.data(), data, sizeof(local), cudaMemcpyDeviceToHost);
    
    size_t i = 0;
    for (auto f : local) {
        std::cout << f << ((i++ % 25 == 24) ? '\n' : ' ');
    }

    std::cout.flush();

    cudaFree(data);
}
