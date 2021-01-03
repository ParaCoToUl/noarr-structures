#include <array>
#include <iostream>

#include "arrr.hpp"

using namespace arrr;
using data_type = decltype(scalar<float> ^ array<'y', 20> ^ array<'x', 20>);

__global__ void use(float *data) {
    data_type as{};
    *(as % fixs<'x','y'>(blockIdx.x, threadIdx.x) % at(data)) = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
    float *data;

    std::array<float, 400> local;

    cudaMalloc(&data, 400*sizeof(float));

    use<<<20, 20>>>(data);

    cudaMemcpy(local.data(), data, 400*sizeof(float), cudaMemcpyDeviceToHost);

    size_t i = 0;
    for (auto f : local) {
        std::cout << f << ' ';
        if (i++ % 20 == 19) std::cout << '\n';
    }

    std::cout.flush();

    cudaFree(data);
}
