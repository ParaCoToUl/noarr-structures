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

template<class... CudaDim>
struct cuda_dims_pack;

using cuda_bx = cuda_dims_pack<cuda_block_x>;
using cuda_bxy = cuda_dims_pack<cuda_block_x, cuda_block_y>;
using cuda_bxyz = cuda_dims_pack<cuda_block_x, cuda_block_y, cuda_block_z>;
using cuda_tx = cuda_dims_pack<cuda_thread_x>;
using cuda_txy = cuda_dims_pack<cuda_thread_x, cuda_thread_y>;
using cuda_txyz = cuda_dims_pack<cuda_thread_x, cuda_thread_y, cuda_thread_z>;

} // namespace helpers

template<char Dim, class T, class CudaDim>
struct cuda_fix_t : contain<T> {
	using base = contain<T>;
	using base::base;

	static constexpr char name[] = "cuda_fix_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>,
		type_param<CudaDim>>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Tuple index must be set statically");
		static_assert(Original::arg_length::is_known, "Index cannot be fixed until its length is set");
		using type = typename Original::ret_sig;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<class State>
	static __device__ inline auto sub_state(State state) noexcept {
		return state.template remove<length_in<Dim>>().template with<index_in<Dim>>(CudaDim::idx());
	}

	template<class State>
	__device__ inline std::size_t size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub, class State>
	__device__ inline std::size_t strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<char QDim, class State>
	__device__ inline std::size_t length(State state) const noexcept {
		static_assert(QDim != Dim, "This dimension is already fixed, it cannot be used from outside");
		return sub_structure().template length<QDim>(sub_state(state));
	}

	template<class Sub, class State>
	__device__ inline auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

namespace helpers {

template<char DimB, char DimT, class CudaDimB, class CudaDimT>
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

template<char... DimsB, char... DimsT, class... CudaDimsB, class... CudaDimsT, class Struct, class Order>
struct cuda_traverser_t<Struct, Order, char_sequence<DimsB...>, char_sequence<DimsT...>, helpers::cuda_dims_pack<CudaDimsB...>, helpers::cuda_dims_pack<CudaDimsT...>> : traverser_t<Struct, Order> {
	using base = traverser_t<Struct, Order>;
	using base::base;

	explicit constexpr cuda_traverser_t(traverser_t<Struct, Order> t) noexcept : base(t) {}

	using base::get_struct;
	using base::get_order;

	using get_fixes = decltype((... ^ helpers::cuda_fix_pair_proto<DimsB, DimsT, CudaDimsB, CudaDimsT>()));

	constexpr dim3 grid_dim() const noexcept {
		auto full = base::top_struct();
		return {(uint)full.template length<DimsB>(empty_state)...};
	}

	constexpr dim3 block_dim() const noexcept {
		auto full = base::top_struct();
		return {(uint)full.template length<DimsT>(empty_state)...};
	}

	explicit constexpr operator bool() const noexcept {
		auto full = base::top_struct();
		return (... && full.template length<DimsT>(empty_state)) && (... && full.template length<DimsB>(empty_state));
	}

	constexpr auto inner() const noexcept
		-> traverser_t<Struct, decltype(get_order() ^ get_fixes())> {
		return traverser_t<Struct, decltype(get_order() ^ get_fixes())>(get_struct(), get_order() ^ get_fixes());
	}


	template<class ...Values>
	constexpr auto simple_run(void kernel(decltype(std::declval<cuda_traverser_t>().inner()), Values...), uint shm_size, Values ...values) const noexcept {
#ifdef __CUDACC__
		kernel<<<grid_dim(), block_dim(), shm_size>>>(inner(), values...);
#else
		static_assert(always_false<cuda_traverser_t>, "This method is only available to CUDA context.");
#endif
	}
};

template<class NewDimsB, class NewDimsT, class NewCudaDimsB, class NewCudaDimsT, class Struct, class Order>
constexpr auto cuda_traverser(traverser_t<Struct, Order> t) noexcept {
	return cuda_traverser_t<Struct, Order, NewDimsB, NewDimsT, NewCudaDimsB, NewCudaDimsT>(t);
}

template<char DimBX, char DimTX, class Struct, class Order>
constexpr auto cuda_threads(traverser_t<Struct, Order> t) noexcept {
	return cuda_traverser<char_sequence<DimBX>, char_sequence<DimTX>, helpers::cuda_bx, helpers::cuda_tx>(t);
}

template<char DimBX, char DimTX, char DimBY, char DimTY, class Struct, class Order>
constexpr auto cuda_threads(traverser_t<Struct, Order> t) noexcept {
	return cuda_traverser<char_sequence<DimBX, DimBY>, char_sequence<DimTX, DimTY>, helpers::cuda_bxy, helpers::cuda_txy>(t);
}

template<char DimBX, char DimTX, char DimBY, char DimTY, char DimBZ, char DimTZ, class Struct, class Order>
constexpr auto cuda_threads(traverser_t<Struct, Order> t) noexcept {
	return cuda_traverser<char_sequence<DimBX, DimBY, DimBZ>, char_sequence<DimTX, DimTY, DimTZ>, helpers::cuda_bxyz, helpers::cuda_txyz>(t);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_CUDA_TRAVERSER_HPP
