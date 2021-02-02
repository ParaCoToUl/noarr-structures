#include <array>
#include <iostream>

#include "noarr_funcs.hpp"

using namespace noarr;

// same body, two data layouts:
template<typename AS>
__global__ void kernel(float *data, AS as) {
    *(float*)((char*)data + (as % fixs<'x','y'>(blockIdx.x, threadIdx.x) % offset())) = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void kernel_handmade(float *data, size_t size) {
    data[blockIdx.x * size + threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
    float *data;

    std::array<float, 400000> local;

    cudaMalloc(&data, sizeof(local));

    // kernel<<<20000, 20>>>(data, (scalar<float> ^ vector<'y'> ^ array<'x', 20000>) % resize<'y'>(20));
    const auto av = array<'y', 20000, vector<'x', scalar<float>>>{};
    const auto avr = av % resize<'x'>(20);
    kernel<<<20000, 20>>>(data, avr);
    kernel_handmade<<<20000, 20>>>(data, 20);

    cudaMemcpy(local.data(), data, sizeof(local), cudaMemcpyDeviceToHost);

    size_t i = 0;
    for (auto f : local) {
        std::cout << f << ((i++ % 20 == 19) ? '\n' : ' ');
    }

    std::cout.flush();

    cudaFree(data);
}
