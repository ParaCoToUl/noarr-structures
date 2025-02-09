#ifndef NOARR_STRUCTURES_STRIDE_ALONG_HPP
#define NOARR_STRUCTURES_STRIDE_ALONG_HPP

#include <cstddef>
#include <type_traits>

#include "../base/state.hpp"
#include "../base/utility.hpp"

#include "../structs/bcast.hpp"
#include "../structs/blocks.hpp"
#include "../structs/layouts.hpp"
#include "../structs/scalar.hpp"
#include "../structs/setters.hpp"
#include "../structs/slice.hpp"
#include "../structs/views.hpp"
#include "../structs/zcurve.hpp"

namespace noarr {

namespace helpers {

// implicitly implements specialization for scalar and all incorrect structures
template<IsDim auto QDim, class T, IsState State>
struct has_stride_along : std::false_type {};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_stride_along<QDim, bcast_t<Dim, T>, State> {
private:
	using Structure = bcast_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == Dim) {
			return State::template contains<length_in<Dim>> && !State::template contains<index_in<Dim>>;
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return make_const<0>(); // stride of a broadcasted dimension is always 0
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_stride_along<QDim, vector_t<Dim, T>, State> {
private:
	using Structure = vector_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == Dim) {
			if constexpr (State::template contains<length_in<Dim>> && !State::template contains<index_in<Dim>>) {
				using namespace constexpr_arithmetic;
				constexpr auto zero = make_const<0>();
				constexpr auto one = make_const<1>();
				return has_offset_of<sub_structure_t, Structure,
				                     decltype(std::declval<State>().template with<index_in<Dim>>(zero))>() &&
				       has_offset_of<sub_structure_t, Structure,
				                     decltype(std::declval<State>().template with<index_in<Dim>>(one))>();
			} else {
				return false;
			}
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			constexpr auto zero = make_const<0>();
			constexpr auto one = make_const<1>();
			return offset_of<sub_structure_t>(structure, state.template with<index_in<Dim>>(one)) -
			       offset_of<sub_structure_t>(structure, state.template with<index_in<Dim>>(zero));
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class IdxT, IsState State>
struct has_stride_along<QDim, fix_t<Dim, T, IdxT>, State> {
private:
	using Structure = fix_t<Dim, T, IdxT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class LenT, IsState State>
struct has_stride_along<QDim, set_length_t<Dim, T, LenT>, State> {
private:
	using Structure = set_length_t<Dim, T, LenT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_stride_along<QDim, hoist_t<Dim, T>, State> {
private:
	using Structure = hoist_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};


template<IsDim auto QDim, class T, auto... DimPairs, IsState State>
requires IsDimPack<decltype(DimPairs)...> && (sizeof...(DimPairs) % 2 == 0)
struct has_stride_along<QDim, rename_t<T, DimPairs...>, State> {
private:
	using Structure = rename_t<T, DimPairs...>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	constexpr static auto QDimNew = helpers::rename_dim<QDim, typename Structure::external, typename Structure::internal>::dim;

	static constexpr bool get_value() noexcept {
		if constexpr (Structure::internal::template contains<QDim> && !Structure::external::template contains<QDim>) {
			return false;
		} else {
			return has_stride_along<QDimNew, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		return has_stride_along<QDimNew, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, class T, auto DimA, auto DimB, auto Dim, IsState State>
requires IsDim<decltype(DimA)> && IsDim<decltype(DimB)> && IsDim<decltype(Dim)> && (DimA != DimB)
struct has_stride_along<QDim, join_t<T, DimA, DimB, Dim>, State> {
private:
	using Structure = join_t<T, DimA, DimB, Dim>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == Dim) {
			return has_stride_along<DimA, sub_structure_t, sub_state_t>::value &&
			       has_stride_along<DimB, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimA || QDim == DimB) {
			return false;
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			return has_stride_along<DimA, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                  structure.sub_state(state)) +
			       has_stride_along<DimB, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, IsState State>
struct has_stride_along<QDim, shift_t<Dim, T, StartT>, State> {
private:
	using Structure = shift_t<Dim, T, StartT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class LenT, IsState State>
struct has_stride_along<QDim, slice_t<Dim, T, StartT, LenT>, State> {
private:
	using Structure = slice_t<Dim, T, StartT, LenT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class EndT, IsState State>
struct has_stride_along<QDim, span_t<Dim, T, StartT, EndT>, State> {
private:
	using Structure = span_t<Dim, T, StartT, EndT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class StrideT, IsState State>
struct has_stride_along<QDim, step_t<Dim, T, StartT, StrideT>, State> {
private:
	using Structure = step_t<Dim, T, StartT, StrideT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state)) * structure.stride();
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_stride_along<QDim, reverse_t<Dim, T>, State> {
private:
	using Structure = reverse_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;

		using sub_stride_t = decltype(+has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state)));

		return -static_cast<std::make_signed_t<sub_stride_t>>(has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(
			structure.sub_structure(), structure.sub_state(state)));
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, class T, IsState State>
requires (DimMajor != DimMinor)
struct has_stride_along<QDim, into_blocks_t<Dim, DimMajor, DimMinor, T>, State> {
private:
	using Structure = into_blocks_t<Dim, DimMajor, DimMinor, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMinor) {
			// the stride of the minor dimension is the stride of Dim in the substructure
			return has_stride_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimMajor) {
			if constexpr (Structure::template has_length<DimMinor, State>()) {
				// the stride of the major dimension is the length of the minor dimension multiplied by the stride of
				// Dim in the substructure
				return has_stride_along<Dim, sub_structure_t, sub_state_t>::value;
			} else {
				// cannot determine the stride of the major dimension
				return false;
			}
		} else if (QDim == Dim) {
			// the Dim dimension is consumed by into_blocks
			return false;
		} else {
			// propagate to the substructure
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

	static constexpr auto get_stride(Structure structure, State state) noexcept {
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_stride_along<Dim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                   structure.sub_state(state));
		} else if constexpr (QDim == DimMajor) {
			const auto minor_len = structure.template length<DimMinor>(state);
			return minor_len *
			       has_stride_along<Dim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                   structure.sub_state(state));
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		return get_stride(structure, state);
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class T, class MinorLenT, IsState State>
requires (DimIsBorder != DimMajor) && (DimIsBorder != DimMinor) && (DimMajor != DimMinor)
struct has_stride_along<QDim, into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, T, MinorLenT>, State> {
private:
	using Structure = into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, T, MinorLenT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMinor) {
			return has_stride_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimMajor) {
			if constexpr (Structure::template has_length<DimMinor, State>()) {
				return has_stride_along<Dim, sub_structure_t, sub_state_t>::value;
			} else {
				return false;
			}
		} else if constexpr (QDim == DimIsBorder || QDim == Dim) {
			return false;
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_stride_along<Dim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                   structure.sub_state(state));
		} else if constexpr (QDim == DimMajor) {
			const auto minor_len = structure.template length<DimMinor>(state);
			return minor_len *
			       has_stride_along<Dim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                   structure.sub_state(state));
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent, class T, IsState State>
requires (DimMajor != DimMinor) && (DimMinor != DimIsPresent) && (DimIsPresent != DimMajor)
struct has_stride_along<QDim, into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, T>, State> {
private:
	using Structure = into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMinor) {
			return has_stride_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimMajor) {
			if constexpr (Structure::template has_length<DimMinor, State>()) {
				return has_stride_along<Dim, sub_structure_t, sub_state_t>::value;
			} else {
				return false;
			}
		} else if constexpr (QDim == DimIsPresent) {
			return true;
		} else if constexpr (QDim == Dim) {
			return false;
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_stride_along<Dim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                   structure.sub_state(state));
		} else if constexpr (QDim == DimMajor) {
			const auto minor_len = structure.template length<DimMinor>(state);
			return minor_len *
			       has_stride_along<Dim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                   structure.sub_state(state));
		} else if constexpr (QDim == DimIsPresent) {
			return make_const<0>();
		} else {
			return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		}
	}
};

} // namespace helpers

template<class T, auto Dim, class State>
concept HasStrideAlong = requires {
	requires IsStruct<T>;
	requires IsState<State>;
	requires IsDim<decltype(Dim)>;

	requires helpers::has_stride_along<Dim, T, State>::value;
};

template<auto Dim, class T, class State>
constexpr auto stride_along(T structure, State state) noexcept
requires HasStrideAlong<T, Dim, State>
{
	return helpers::has_stride_along<Dim, T, State>::stride(structure, state);
}

// tests

// bcast_t
static_assert(HasStrideAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ bcast<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) == 0);
static_assert(!HasStrideAlong<bcast_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!HasStrideAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasStrideAlong<bcast_t<'x', scalar<int>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

// vector_t
static_assert(HasStrideAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) ==
              sizeof(int));
static_assert(!HasStrideAlong<vector_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!HasStrideAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasStrideAlong<vector_t<'x', scalar<int>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(
	HasStrideAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(stride_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
                                state<state_item<length_in<'y'>, std::size_t>>(42)) == sizeof(int));
static_assert(
	!HasStrideAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(HasStrideAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y',
                             state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(stride_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
                                state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>(
									42, 42)) == sizeof(int));
static_assert(HasStrideAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
                             state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
                                state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>(
									5, 42)) == 42 * sizeof(int));

// scalar
static_assert(!HasStrideAlong<scalar<int>, 'x', state<>>);
static_assert(!HasStrideAlong<scalar<int>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<scalar<int>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<scalar<int>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

// fix_t
static_assert(!HasStrideAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(
	!HasStrideAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<fix_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<fix_t<'x', vector_t<'y', scalar<int>>, std::size_t>, 'y',
                             state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(stride_along<'y'>(scalar<int>() ^ vector<'y'>() ^ fix<'x'>(0),
                                state<state_item<length_in<'y'>, std::size_t>>(0)) == sizeof(int));

// set_length_t
static_assert(!HasStrideAlong<set_length_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(HasStrideAlong<set_length_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
static_assert(stride_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ set_length<'x'>(0), state<>()) == 0);
static_assert(HasStrideAlong<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(1), state<>()) == sizeof(int));

// rename_t
static_assert(!HasStrideAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<>>);
static_assert(!HasStrideAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(HasStrideAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(stride_along<'y'>(scalar<int>() ^ vector<'x'>() ^ rename<'x', 'y'>(),
								state<state_item<length_in<'y'>, std::size_t>>(42)) == sizeof(int));

// join_t
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x',
							  state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<index_in<'y'>, std::size_t>>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y',
							  state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<>>);
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<index_in<'z'>, std::size_t>>>);
static_assert(HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<length_in<'z'>, std::size_t>>>);
static_assert(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ vector<'y'>() ^ join<'x', 'y', 'z'>(),
								state<state_item<length_in<'z'>, std::size_t>>(42)) == sizeof(int) + 42 * sizeof(int));
static_assert(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z',
							  state<state_item<length_in<'z'>, std::size_t>, state_item<index_in<'z'>, std::size_t>>>);

// shift_t
static_assert(!HasStrideAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(!HasStrideAlong<shift_t<'x', scalar<int>, std::size_t>, 'x',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<shift_t<'x', scalar<int>, std::size_t>, 'x',
							  state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ shift<'x'>(4), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
static_assert(HasStrideAlong<shift_t<'y', vector_t<'x', scalar<int>>, std::size_t>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ shift<'y'>(4), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
static_assert(!HasStrideAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'y',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<shift_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'x',
							 state<>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ shift<'x'>(4), state<>()) == sizeof(int));
static_assert(!HasStrideAlong<shift_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'y',
							  state<>>);

// slice_t
static_assert(!HasStrideAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasStrideAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
							  state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ slice<'x'>(4, 2), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
static_assert(HasStrideAlong<slice_t<'y', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ slice<'y'>(4, 3), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
static_assert(!HasStrideAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'y',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<slice_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x',
							 state<>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ slice<'x'>(4, 4), state<>()) == sizeof(int));
static_assert(!HasStrideAlong<slice_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'y',
							  state<>>);

// span_t
static_assert(!HasStrideAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasStrideAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
							  state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ span<'x'>(4, 6), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
static_assert(HasStrideAlong<span_t<'y', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ span<'y'>(4, 8), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
static_assert(!HasStrideAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'y',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<span_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x',
							 state<>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ span<'x'>(4, 7), state<>()) == sizeof(int));
static_assert(!HasStrideAlong<span_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'y',
							  state<>>);

// step_t
static_assert(!HasStrideAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasStrideAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
							  state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'x'>(4, 6), state<state_item<length_in<'x'>, std::size_t>>(42)) == 6 * sizeof(int));
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'x'>(4, 6) ^ step<'x'>(0, 2), state<state_item<length_in<'x'>, std::size_t>>(42)) == 2 * 6 * sizeof(int));
static_assert(HasStrideAlong<step_t<'y', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'y'>(4, 8), state<state_item<length_in<'x'>, std::size_t>>(42)) == 8 * sizeof(int));
static_assert(!HasStrideAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'y',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<step_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x',
							 state<>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ step<'x'>(4, 7), state<>()) == 7 * sizeof(int));
static_assert(!HasStrideAlong<step_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'y',
							  state<>>);

// reverse_t
static_assert(!HasStrideAlong<reverse_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!HasStrideAlong<reverse_t<'x', scalar<int>>, 'x',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<reverse_t<'x', scalar<int>>, 'x',
							  state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) == -static_cast<int>(sizeof(int)));
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'x'>() ^ reverse<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) == static_cast<int>(sizeof(int)));
static_assert(HasStrideAlong<reverse_t<'y', vector_t<'x', scalar<int>>>, 'x',
							 state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'y'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) == -static_cast<int>(sizeof(int)));
static_assert(!HasStrideAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'y',
							  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<reverse_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x',
							 state<>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ reverse<'x'>(), state<>()) == -static_cast<int>(sizeof(int)));
static_assert(!HasStrideAlong<reverse_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y',
							  state<>>);

// into_blocks_t
static_assert(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<>>);
static_assert(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<>>);
static_assert(
	!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);

static_assert(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x', state<>>);
static_assert(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y', state<>>);
static_assert(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
                              state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
                             state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'y'>(scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'x', 'y'>(),
                                state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>(
									42, 5)) == sizeof(int));
static_assert(HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
                             state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'x', 'y'>(),
                                state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>(
									42, 5)) == 42 * sizeof(int));

// into_blocks_static_t
static_assert(!HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'x', state<state_item<index_in<'y'>, lit_t<0>>>>);
static_assert(HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'z', state<state_item<index_in<'y'>, lit_t<0>>>>);
static_assert(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>),
state<state_item<index_in<'y'>, lit_t<0>>>()) == 4 * sizeof(int));
static_assert(HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'w', state<state_item<index_in<'y'>, lit_t<0>>>>);
static_assert(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>),
state<state_item<index_in<'y'>, lit_t<0>>>()) == sizeof(int));
static_assert(!HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'y', state<>>);

static_assert(HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'z', state<state_item<index_in<'y'>, lit_t<1>>>>);
static_assert(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>),
state<state_item<index_in<'y'>, lit_t<1>>>()) == 2 * sizeof(int));
static_assert(HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'w', state<state_item<index_in<'y'>, lit_t<1>>>>);
static_assert(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>),
state<state_item<index_in<'y'>, lit_t<1>>>()) == sizeof(int));

// into_blocks_dynamic_t
static_assert(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<>>);
static_assert(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<>>);
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<>>);
static_assert(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<>()) == sizeof(int));
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<>>);
static_assert(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<>()) == 0);

static_assert(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'x'>, std::size_t>>(7)) == sizeof(int));
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'x'>, std::size_t>>(7)) == 0);

static_assert(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(stride_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'y'>, std::size_t>>(7)) == 6 * sizeof(int));
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'y'>, std::size_t>>(7)) == sizeof(int));
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'y'>, std::size_t>>(7)) == 0);

static_assert(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<state_item<length_in<'z'>, std::size_t>>>);
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<state_item<length_in<'z'>, std::size_t>>>);
static_assert(stride_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'z'>, std::size_t>>(7)) == 7 * sizeof(int));
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<state_item<length_in<'z'>, std::size_t>>>);
static_assert(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'z'>, std::size_t>>(7)) == sizeof(int));
static_assert(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<state_item<length_in<'z'>, std::size_t>>>);
static_assert(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'z'>, std::size_t>>(7)) == 0);

} // namespace noarr

#endif // NOARR_STRUCTURES_STRIDE_ALONG_HPP
