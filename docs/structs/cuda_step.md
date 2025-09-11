# cuda_step

Split the structure among CUDA threads according to the specified [dimension](../Glossary.md#dimension).

```hpp
#include <noarr/structures/interop/cuda_step.cuh>

template<typename CG>
__device__ inline proto noarr::cuda_step();

__device__ inline proto noarr::cuda_step(const auto &cg);

__device__ inline proto noarr::cuda_step_block();

__device__ inline proto noarr::cuda_step_grid();

template<auto Dim, typename CG>
__device__ inline proto noarr::cuda_step();

template<auto Dim>
__device__ inline proto noarr::cuda_step(const auto &cg);

template<auto Dim>
__device__ inline proto noarr::cuda_step_block();

template<auto Dim>
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
See [usage examples](#usage-examples) for more information.

These functions are implemented shortcuts for [`noarr::step`](step.md).
`CG::thread_rank()` or `cg.thread_rank()` is used for `step`'s `start` parameter,
and `CG::num_threads()` or `cg.num_threads()` is used for the `step` parameter.


## Usage examples

Use `cuda_step` to traverse a structure using all threads of the specified cooperative group.
For example, a CUDA grid executing the following code traverses a vector regardless of the kernel launch parameters (grid dim, block dim).

```cpp
auto structure = noarr::scalar<float>() ^ noarr::vector<'i'>(1024*1024);

noarr::traverser(structure).order(noarr::cuda_step_grid()).for_each([&](auto state) {
	std::size_t off = structure | noarr::offset(state); // or use bag
	// ...
});
```

If there are 4096 threads (e.g. 64 blocks of 64), thread 0 will process elements 0, 4096, 8192, 12288, etc, thread 1 will process elements 1, 4097, 8193, 12289, etc, etc.

Replace `cuda_step_grid` with `cuda_step_block` if each block has its own `structure` to traverse.
Thread 0 will process elements 0, 64, 128, 192, etc, thread 1 will process elements 1, 65, 129, 193, etc, etc.

The shortcut `noarr::cuda_step_grid()` is equivalent to `noarr::cuda_step<cooperative_groups::grid_group>()` and `noarr::cuda_step(cooperative_groups::grid_group())`.
The shortcut `noarr::cuda_step_block()` is equivalent to `noarr::cuda_step<cooperative_groups::thread_block>()` and `noarr::cuda_step(cooperative_groups::thread_block())`.
For other examples of cooperative groups and ways to create them, see [the Nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups).

### Explicitly specifying a dimension

When the structure has more dimensions, [`noarr::step`](step.md) chooses the outer-most dimension.
So does `noarr::cuda_step`, for which it is usually not the best choice. For example:

```cpp
auto matrix = noarr::scalar<float>() ^ noarr::vector<'j'>(1024) ^ noarr::vector<'i'>(1024);

noarr::traverser(matrix).order(noarr::cuda_step_block()).for_each([&](auto state) {
	std::size_t off = matrix | noarr::offset(state); // or use bag
	// ...
});
```

Assuming a block of 64 threads again, thread 0 will process the whole structure at `i=0`, `i=64`, etc, thread 1 will process the whole structure at `i=1`, `i=65`, etc, etc.
All threads will start at element `i=?, j=0` and therefore simultaneously access elements that are far apart (if in global memory) and in the same memory bank (if in shared memory).

To avoid this, either of the following is enough (`Dim` is stands for the innermost dimension, `'j'` in the example):
- use `noarr::cuda_step_block<Dim>()` (or analogically with `_grid` or another cooperative group)
- use `noarr::hoist<Dim>()` before applying `noarr::cuda_striped[_*]` - this is useful if you do not know the inner-most dimension or want to define it somewhere else

```cpp
noarr::traverser(matrix).order(noarr::cuda_step_block<'j'>()).for_each([&](auto state) {
	// ...
});
```

Additionally, you can use [`noarr::merge_blocks`](merge_blocks.md) to convert two dimensions to one (note: this name is unrelated to CUDA thread blocks).
This can be useful if the number of threads exceeds the length in the dimension (which would otherwise mean some threads would have nothing to do).

```cpp
noarr::traverser(matrix).order(noarr::merge_blocks<'i', 'j', 't'>() ^ noarr::cuda_step_grid<'t'>()).for_each([&](auto state) {
	// ...
});
```
