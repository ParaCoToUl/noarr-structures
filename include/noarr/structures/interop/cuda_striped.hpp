#ifndef NOARR_STRUCTURES_CUDA_STRIPED_HPP
#define NOARR_STRUCTURES_CUDA_STRIPED_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../extra/struct_traits.hpp"
#include "../structs/scalar.hpp"

namespace noarr {

constexpr std::size_t cuda_shm_bank_size = 4;
constexpr std::size_t cuda_shm_bank_throughput = 2;
constexpr std::size_t cuda_shm_bank_count = 32;
constexpr std::size_t cuda_striped_default_period = cuda_shm_bank_size * cuda_shm_bank_throughput * cuda_shm_bank_count;

// Tag for use in state
struct cuda_stripe_index;

template<std::size_t NumStripes, class ElemType, std::size_t Period, std::size_t BankAlignment, class T>
struct cuda_striped_t : contain<T> {
	static_assert(is_struct<ElemType>(), "The element type of cuda_striped must be a noarr structure.");

	static constexpr char name[] = "cuda_striped_t";
	using params = struct_params<
		value_param<std::size_t, NumStripes>,
		structure_param<ElemType>,
		value_param<std::size_t, Period>,
		value_param<std::size_t, BankAlignment>,
		structure_param<T>>;

	constexpr cuda_striped_t() noexcept = default;
	explicit constexpr cuda_striped_t(T sub_structure) noexcept : contain<T>(sub_structure) {}

	constexpr T sub_structure() const noexcept { return contain<T>::template get<0>(); }
	static constexpr std::size_t elem_size = decltype(std::declval<ElemType>().size(state<>()))::value;
	static constexpr std::size_t stripe_padded_width = (Period / (NumStripes*BankAlignment)) * BankAlignment;
	static constexpr std::size_t stripe_width_elems = stripe_padded_width / elem_size;
	static constexpr std::size_t stripe_width = stripe_width_elems * elem_size;
	static_assert(stripe_width_elems, "Cannot fit enough stripes into separate banks");

	using signature = typename T::signature;

	template<class State>
	constexpr auto size(State state) const noexcept {
		using namespace constexpr_arithmetic;
		// substructure size
		auto sub_size = sub_structure().size(state.template remove<cuda_stripe_index>());
		// total elements in sub-structure = total elements in each stripe
		auto sub_elements = sub_size / make_const<elem_size>();
		// stripe length = ceil(total elements in stripe / total elements in stripe width)
		auto stripe_len = (sub_elements + make_const<stripe_width_elems - 1>()) / make_const<stripe_width_elems>();
		// total size = stripe length * (padded stripe width * num stripes + padding up to period) = stripe length * period
		return stripe_len * make_const<Period>();
	}

	template<class Sub, class State>
	constexpr auto strict_offset_of(State state) const noexcept {
		using namespace constexpr_arithmetic;
		auto sub_offset = offset_of<Sub>(sub_structure(), state.template remove<cuda_stripe_index>());
		auto offset_major = sub_offset / make_const<stripe_width>();
		if constexpr(std::is_same_v<Sub, ElemType> && stripe_width_elems == 1) {
			// Optimization: offset_minor should be zero.
			return offset_inner(state, offset_major);
		} else {
			auto offset_minor = sub_offset % make_const<stripe_width>();
			return offset_inner(state, offset_major) + offset_minor;
		}
	}

	template<char QDim, class State>
	constexpr auto length(State state) const noexcept {
		return sub_structure().template length<QDim>(state.template remove<cuda_stripe_index>());
	}

	template<class Sub, class State>
	constexpr void strict_state_at(State) const noexcept {
		static_assert(always_false<cuda_striped_t>, "A cuda_striped_t cannot be used in this context");
	}

private:
	template<class State, class Idx>
	constexpr auto offset_inner(State state, Idx index_of_period) const noexcept {
		using namespace constexpr_arithmetic;
		auto offset_of_period = index_of_period * make_const<Period>();
		if constexpr(State::template contains<cuda_stripe_index>) {
			auto offset_of_stripe = state.template get<cuda_stripe_index>() * make_const<stripe_padded_width>();
			return offset_of_period + offset_of_stripe;
		} else {
			std::size_t offset_of_stripe = (threadIdx.x % NumStripes) * stripe_padded_width;
			return offset_of_period + offset_of_stripe;
		}
	}
};

template<std::size_t NumStripes, class ElemType, std::size_t Period, std::size_t BankAlignment>
struct cuda_striped_proto {
	static_assert(is_struct<ElemType>(), "The element type of cuda_striped must be a noarr structure. Omit the type to imply scalar<...>, or specify scalar<...> (or any other noarr structure) explicitly.");
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return cuda_striped_t<NumStripes, ElemType, Period, BankAlignment, Struct>(s); }
};

template<std::size_t NumStripes, class ElemType, std::size_t Period = noarr::cuda_striped_default_period, std::size_t BankAlignment = noarr::cuda_shm_bank_size>
constexpr auto cuda_striped() noexcept { return cuda_striped_proto<NumStripes, ElemType, Period, BankAlignment>(); }



template<std::size_t NumStripes, std::size_t Period, std::size_t BankAlignment, class T>
using cuda_scalar_striped_t = cuda_striped_t<NumStripes, scalar<unchecked_scalar_t<T>>, Period, BankAlignment, T>;

template<std::size_t NumStripes, std::size_t Period, std::size_t BankAlignment>
struct cuda_scalar_striped_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return cuda_scalar_striped_t<NumStripes, Period, BankAlignment, Struct>(s); }
};

template<std::size_t NumStripes, std::size_t Period = noarr::cuda_striped_default_period, std::size_t BankAlignment = noarr::cuda_shm_bank_size>
constexpr auto cuda_striped() noexcept { return cuda_scalar_striped_proto<NumStripes, Period, BankAlignment>(); }

} // namespace noarr

#endif // NOARR_STRUCTURES_CUDA_STRIPED_HPP
