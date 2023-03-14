# cuda_step

Split the structure among CUDA threads according to the specified [dimension](../Glossary.md#dimension).

```hpp
#include <noarr/structures/interop/cuda_step.cuh>

template<typename CG>
__device__ inline proto noarr::cuda_step();

__device__ inline proto noarr::cuda_step(const auto &cg);

__device__ inline proto noarr::cuda_step_block();

__device__ inline proto noarr::cuda_step_grid();

template<char Dim, typename CG>
__device__ inline proto noarr::cuda_step();

template<char Dim>
__device__ inline proto noarr::cuda_step(const auto &cg);

template<char Dim>
__device__ inline proto noarr::cuda_step_block();

template<char Dim>
__device__ inline proto noarr::cuda_step_grid();
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

These functions create a proto-structure that splits a structure more or less evenly among a group of CUDA threads.
The group can be specified as a [cooperative group](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups),
but shortcuts for "all threads in current block" and "all threads in current grid" are available.

The structure is split according to one selected dimension in a round-robin fashion.
By default, the outer-most dimension is chosen, which might not be what you want.
Unless there is another mechanism in place to reorder the dimensions as necessary,
you should specify the `Dim` template parameter explicitly to the inner-most dimension.

These functions are implemented shortcuts for [`noarr::step`](step.md).
`CG::thread_rank()` or `cg.thread_rank()` is used for `step`'s `start` parameter,
and `CG::num_threads()` or `cg.num_threads()` is used for the `step` parameter.
