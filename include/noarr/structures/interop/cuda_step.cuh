#ifndef NOARR_STRUCTURES_CUDA_STEP_HPP
#define NOARR_STRUCTURES_CUDA_STEP_HPP

#include <cooperative_groups.h>

#include "../structs/slice.hpp"

namespace noarr {

template<char Dim, class CG>
__device__ inline auto cuda_step(const CG &cg) noexcept {
	return step<Dim>(cg.thread_rank(), cg.num_threads());
}

template<char Dim, class CG>
__device__ inline auto cuda_step() noexcept {
	return step<Dim>(CG::thread_rank(), CG::num_threads());
}

template<char Dim>
__device__ inline auto cuda_step_block() noexcept {
	return cuda_step<Dim, ::cooperative_groups::thread_block>();
}

template<char Dim>
__device__ inline auto cuda_step_grid() noexcept {
	return cuda_step<Dim, ::cooperative_groups::grid_group>();
}

template<class CG>
__device__ inline auto cuda_auto_step(const CG &cg) noexcept {
	return auto_step(cg.thread_rank(), cg.num_threads());
}

template<class CG>
__device__ inline auto cuda_auto_step() noexcept {
	return auto_step(CG::thread_rank(), CG::num_threads());
}

__device__ inline auto cuda_auto_step_block() noexcept {
	return cuda_auto_step<::cooperative_groups::thread_block>();
}

__device__ inline auto cuda_auto_step_grid() noexcept {
	return cuda_auto_step<::cooperative_groups::grid_group>();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_CUDA_STEP_HPP
