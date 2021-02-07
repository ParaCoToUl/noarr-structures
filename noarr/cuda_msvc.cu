#include <array>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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
    float *data;
    std::cout << "hi" << std::endl;

    std::array<float, 40000> local;
    

    cudaMalloc(&data, sizeof(local));

    // kernel<<<20000, 20>>>(data, (scalar<float> ^ vector<'y'> ^ array<'x', 20000>) % resize<'y'>(20));
    const auto av = array<'y', 2000, vector<'x', scalar<float>>>{};
    volatile std::size_t s = 20;
    const auto avr = av % resize<'x'>(s);
    kernel<<<2000, 20>>>(data, avr);
    //kernel_handmade<<<2000, 20>>>(data, 20);
    
    cudaMemcpy(local.data(), data, sizeof(local), cudaMemcpyDeviceToHost);
    
    size_t i = 0;
    for (auto f : local) {
        std::cout << f << ((i++ % 25 == 24) ? '\n' : ' ');
    }

    std::cout.flush();

    cudaFree(data);
}
