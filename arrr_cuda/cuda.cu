#include <array>
#include <iostream>

#include "arrr.hpp"

using namespace arrr;

template<typename AS>
__global__ void use(float *data, AS as) {
    *(as % fixs<'x','y'>(blockIdx.x, threadIdx.x) % at(data)) = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
    float *data;

    std::array<float, 400000> local;

    cudaMalloc(&data, sizeof(local));

    use<<<20000, 20>>>(data, (scalar<float> ^ vector<'y'> ^ array<'x', 20000>) % resize<'y'>(20));

    cudaMemcpy(local.data(), data, sizeof(local), cudaMemcpyDeviceToHost);

    size_t i = 0;
    for (auto f : local) {
        std::cout << f << ' ';
        if (i++ % 20 == 19) std::cout << '\n';
    }

    std::cout.flush();

    cudaFree(data);
}
