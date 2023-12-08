#ifndef NOARR_STRUCTURES_CUDA_TRAVERSER_HPP
#define NOARR_STRUCTURES_CUDA_TRAVERSER_HPP

#include "../extra/traverser.hpp"

namespace noarr {

namespace helpers {

struct cuda_block_x  { static __device__ inline auto idx() noexcept { return blockIdx.x; } };
struct cuda_block_y  { static __device__ inline auto idx() noexcept { return blockIdx.y; } };
struct cuda_block_z  { static __device__ inline auto idx() noexcept { return blockIdx.z; } };
struct cuda_thread_x { static __device__ inline auto idx() noexcept { return threadIdx.x; } };
struct cuda_thread_y { static __device__ inline auto idx() noexcept { return threadIdx.y; } };
struct cuda_thread_z { static __device__ inline auto idx() noexcept { return threadIdx.z; } };

template<class ...CudaDims>
struct cuda_dims_pack;

using cuda_bx = cuda_dims_pack<cuda_block_x>;
using cuda_bxy = cuda_dims_pack<cuda_block_x, cuda_block_y>;
using cuda_bxyz = cuda_dims_pack<cuda_block_x, cuda_block_y, cuda_block_z>;
using cuda_tx = cuda_dims_pack<cuda_thread_x>;
using cuda_txy = cuda_dims_pack<cuda_thread_x, cuda_thread_y>;
using cuda_txyz = cuda_dims_pack<cuda_thread_x, cuda_thread_y, cuda_thread_z>;

} // namespace helpers

template<IsDim auto Dim, class T, class CudaDim>
struct cuda_fix_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "cuda_fix_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>,
		type_param<CudaDim>>;

	constexpr T sub_structure() const noexcept { return this->get(); }

private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Tuple index must be set statically");
		static_assert(Original::arg_length::is_known, "Index cannot be fixed until its length is set");
		using type = typename Original::ret_sig;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	static __device__ inline auto sub_state(IsState auto state) noexcept {
		return state.template remove<length_in<Dim>>().template with<index_in<Dim>>(CudaDim::idx());
	}

	__device__ inline std::size_t size(IsState auto state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub>
	__device__ inline std::size_t strict_offset_of(IsState auto state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<IsDim auto QDim>
	__device__ inline std::size_t length(IsState auto state) const noexcept {
		static_assert(QDim != Dim, "This dimension is already fixed, it cannot be used from outside");
		return sub_structure().template length<QDim>(sub_state(state));
	}

	template<class Sub>
	__device__ inline auto strict_state_at(IsState auto state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

namespace helpers {

template<IsDim auto DimB, IsDim auto DimT, class CudaDimB, class CudaDimT>
struct cuda_fix_pair_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		using fixt = cuda_fix_t<DimT, Struct, CudaDimT>;
		using fixb = cuda_fix_t<DimB, fixt, CudaDimB>;
		return fixb(fixt(s));
	}
};

} // namespace helpers

template<class Struct, class Order, class DimsB, class DimsT, class CudaDimsB, class CudaDimsT>
struct cuda_traverser_t;

template<auto ...DimsB, auto ...DimsT, class ...CudaDimsB, class ...CudaDimsT, class Struct, class Order>
struct cuda_traverser_t<Struct, Order, dim_sequence<DimsB...>, dim_sequence<DimsT...>, helpers::cuda_dims_pack<CudaDimsB...>, helpers::cuda_dims_pack<CudaDimsT...>> : traverser_t<Struct, Order> {
	using base = traverser_t<Struct, Order>;
	using base::base;

	explicit constexpr cuda_traverser_t(traverser_t<Struct, Order> t) noexcept : base(t) {}

	using base::get_struct;
	using base::get_order;

	using get_fixes = decltype((... ^ helpers::cuda_fix_pair_proto<DimsB, DimsT, CudaDimsB, CudaDimsT>()));

	constexpr dim3 grid_dim() const noexcept {
		const auto full = this->top_struct();
		return {(uint)full.template length<DimsB>(empty_state)...};
	}

	constexpr dim3 block_dim() const noexcept {
		const auto full = this->top_struct();
		return {(uint)full.template length<DimsT>(empty_state)...};
	}

	explicit constexpr operator bool() const noexcept {
		const auto full = this->top_struct();
		return (... && full.template length<DimsT>(empty_state)) && (... && full.template length<DimsB>(empty_state));
	}

	constexpr auto inner() const noexcept
		-> traverser_t<Struct, decltype(get_order() ^ get_fixes())> {
		return traverser_t<Struct, decltype(get_order() ^ get_fixes())>(get_struct(), get_order() ^ get_fixes());
	}


#ifdef __CUDACC__
	template<class ...Values>
	constexpr auto simple_run(void kernel(traverser_t<Struct, decltype(std::declval<base>().get_order() ^ get_fixes())>, Values...), uint shm_size, Values ...values) const noexcept {
		kernel<<<grid_dim(), block_dim(), shm_size>>>(inner(), values...);
	}
#endif
};

template<class NewDimsB, class NewDimsT, class NewCudaDimsB, class NewCudaDimsT, class Struct, class Order>
constexpr auto cuda_traverser(traverser_t<Struct, Order> t) noexcept {
	return cuda_traverser_t<Struct, Order, NewDimsB, NewDimsT, NewCudaDimsB, NewCudaDimsT>(t);
}

template<IsDim auto DimBX, IsDim auto DimTX, class Struct, class Order>
constexpr auto cuda_threads(traverser_t<Struct, Order> t) noexcept {
	return cuda_traverser<dim_sequence<DimBX>, dim_sequence<DimTX>, helpers::cuda_bx, helpers::cuda_tx>(t);
}

template<IsDim auto DimBX, IsDim auto DimTX, IsDim auto DimBY, IsDim auto DimTY, class Struct, class Order>
constexpr auto cuda_threads(traverser_t<Struct, Order> t) noexcept {
	return cuda_traverser<dim_sequence<DimBX, DimBY>, dim_sequence<DimTX, DimTY>, helpers::cuda_bxy, helpers::cuda_txy>(t);
}

template<IsDim auto DimBX, IsDim auto DimTX, IsDim auto DimBY, IsDim auto DimTY, IsDim auto DimBZ, IsDim auto DimTZ, class Struct, class Order>
constexpr auto cuda_threads(traverser_t<Struct, Order> t) noexcept {
	return cuda_traverser<dim_sequence<DimBX, DimBY, DimBZ>, dim_sequence<DimTX, DimTY, DimTZ>, helpers::cuda_bxyz, helpers::cuda_txyz>(t);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_CUDA_TRAVERSER_HPP
