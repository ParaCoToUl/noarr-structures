#ifndef NOARR_STRUCTURES_OFFSET_ALONG_HPP
#define NOARR_STRUCTURES_OFFSET_ALONG_HPP

#include <cstddef>
#include <type_traits>

#include "../base/state.hpp"

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
struct has_offset_along : std::false_type {};


template<IsDim auto QDim, class T, IsState State>
struct generic_has_offset_along {
private:
	using Structure = T;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
		                                                                  structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, bcast_t<Dim, T>, State> {
private:
	using Structure = bcast_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == Dim) {
			return has_offset_of<sub_structure_t, Structure, State>();
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			return make_const<0>(); // offset of a broadcasted dimension is always 0
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, vector_t<Dim, T>, State> {
private:
	using Structure = vector_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == Dim) {
			return has_offset_of<sub_structure_t, Structure, State>();
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			const auto index = state.template get<index_in<Dim>>();
			return index * structure.sub_structure().size(structure.sub_state(state));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		}
	}
};

// tuple_t
template<IsDim auto QDim, IsDim auto Dim, class... Ts, IsState State>
struct has_offset_along<QDim, tuple_t<Dim, Ts...>, State> {
private:
	using Structure = tuple_t<Dim, Ts...>;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (State::template contains<index_in<Dim>>) {
			using index_t = state_get_t<State, index_in<Dim>>;

			if constexpr (requires { index_t::value; requires (index_t::value < sizeof...(Ts)); }) {
				constexpr std::size_t index = state_get_t<State, index_in<Dim>>::value;

				using sub_structure_t = typename Structure::template sub_structure_t<index>;

				if constexpr (QDim == Dim) {
					return has_offset_of<sub_structure_t, Structure, State>();
				} else {
					return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
				}
			} else {
				return false;
			}
		} else {
			return false;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		constexpr std::size_t index = state_get_t<State, index_in<Dim>>::value;

		using sub_structure_t = typename Structure::template sub_structure_t<index>;

		if constexpr (QDim == Dim) {
			return offset_of<sub_structure_t>(structure, state);
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(index),
																		structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class IdxT, IsState State>
struct has_offset_along<QDim, fix_t<Dim, T, IdxT>, State> : generic_has_offset_along<QDim, fix_t<Dim, T, IdxT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, class LenT, IsState State>
struct has_offset_along<QDim, set_length_t<Dim, T, LenT>, State> : generic_has_offset_along<QDim, set_length_t<Dim, T, LenT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, hoist_t<Dim, T>, State> : generic_has_offset_along<QDim, hoist_t<Dim, T>, State> {};

template<IsDim auto QDim, class T, auto... DimPairs, IsState State>
requires IsDimPack<decltype(DimPairs)...> && (sizeof...(DimPairs) % 2 == 0)
struct has_offset_along<QDim, rename_t<T, DimPairs...>, State> {
private:
	using Structure = rename_t<T, DimPairs...>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	constexpr static auto QDimNew = helpers::rename_dim<QDim, typename Structure::external, typename Structure::internal>::dim;

	static constexpr bool get_value() noexcept {
		return has_offset_along<QDimNew, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (Structure::internal::template contains<QDim> && !Structure::external::template contains<QDim>) {
			return false;
		} else {
			return has_offset_along<QDimNew, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
																				structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, class T, auto DimA, auto DimB, auto Dim, IsState State>
requires IsDim<decltype(DimA)> && IsDim<decltype(DimB)> && IsDim<decltype(Dim)> && (DimA != DimB)
struct has_offset_along<QDim, join_t<T, DimA, DimB, Dim>, State> {
private:
	using Structure = join_t<T, DimA, DimB, Dim>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == Dim) {
			return has_offset_along<DimA, sub_structure_t, sub_state_t>::value && has_offset_along<DimB, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimA || QDim == DimB) {
			return false;
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			return has_offset_along<DimA, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state))
				+ has_offset_along<DimB, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, IsState State>
struct has_offset_along<QDim, shift_t<Dim, T, StartT>, State> : generic_has_offset_along<QDim, shift_t<Dim, T, StartT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class LenT, IsState State>
struct has_offset_along<QDim, slice_t<Dim, T, StartT, LenT>, State> : generic_has_offset_along<QDim, slice_t<Dim, T, StartT, LenT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class EndT, IsState State>
struct has_offset_along<QDim, span_t<Dim, T, StartT, EndT>, State> : generic_has_offset_along<QDim, span_t<Dim, T, StartT, EndT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class StrideT, IsState State>
struct has_offset_along<QDim, step_t<Dim, T, StartT, StrideT>, State> : generic_has_offset_along<QDim, step_t<Dim, T, StartT, StrideT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, reverse_t<Dim, T>, State> : generic_has_offset_along<QDim, reverse_t<Dim, T>, State> {};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, class T, IsState State>
struct has_offset_along<QDim, into_blocks_t<Dim, DimMajor, DimMinor, T>, State> {
private:
	using Structure = into_blocks_t<Dim, DimMajor, DimMinor, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMajor) {
			return has_offset_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimMinor) {
			return has_offset_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == Dim) {
			return false;
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMajor) {
			return offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMinor>>(make_const<0>()))) -
			offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimMinor) {
			return offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>>(make_const<0>()))) -
			offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(make_const<0>(), make_const<0>())));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class T, class MinorLenT, IsState State>
requires (DimIsBorder != DimMajor) && (DimIsBorder != DimMinor) && (DimMajor != DimMinor)
struct has_offset_along<QDim, into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, T, MinorLenT>, State> {
private:
	using Structure = into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, T, MinorLenT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMajor) {
			return has_offset_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimMinor) {
			return has_offset_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimIsBorder) {
			return has_offset_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == Dim) {
			return false;
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;

		if constexpr (QDim == DimMajor) {
			return offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMinor>>(make_const<0>()))) -
			offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimMinor) {
			return offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>>(make_const<0>()))) -
			offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimIsBorder) {
			return offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(make_const<0>(), make_const<0>())));
			- offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>, index_in<DimIsBorder>>(make_const<0>(), make_const<0>(), make_const<0>())));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent, class T, IsState State>
requires (DimMajor != DimMinor) && (DimMinor != DimIsPresent) && (DimIsPresent != DimMajor)
struct has_offset_along<QDim, into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, T>, State> {
private:
	using Structure = into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMajor) {
			return has_offset_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimMinor) {
			return has_offset_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimIsPresent) {
			return has_offset_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == Dim) {
			return false;
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;

		if constexpr (QDim == DimMajor) {
			return offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMinor>>(make_const<0>())))
			- offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimMinor) {
			return offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>>(make_const<0>())))
			- offset_along<Dim>(structure.sub_structure(), structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimIsPresent) {
			return make_const<0>();
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, merge_blocks_t<DimMajor, DimMinor, Dim, T>, State> {
private:
	using Structure = merge_blocks_t<DimMajor, DimMinor, Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == Dim) {
			return has_offset_along<DimMajor, sub_structure_t, sub_state_t>::value && has_offset_along<DimMinor, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimMajor || QDim == DimMinor) {
			return false;
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			return has_offset_along<DimMajor, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state))
				+ has_offset_along<DimMinor, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state));
		}
	}
};

} // namespace helpers

template<class T, auto Dim, class State>
concept HasOffsetAlong = requires {
	requires IsStruct<T>;
	requires IsState<State>;
	requires IsDim<decltype(Dim)>;

	requires helpers::has_offset_along<Dim, T, State>::value;
};

template<auto Dim, class T, class State>
constexpr auto offset_along(T structure, State state) noexcept
requires HasOffsetAlong<T, Dim, State>
{
	return helpers::has_offset_along<Dim, T, State>::offset(structure, state);
}

// tests

// bcast_t
static_assert(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
                                     state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 0 * sizeof(int));
static_assert(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
									 state<state_item<index_in<'x'>, std::size_t>>(5)) == 0 * sizeof(int));
static_assert(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
									 state<state_item<index_in<'x'>, std::size_t>>(5)) == 0 * sizeof(int));
static_assert(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
									 state<state_item<length_in<'y'>, std::size_t>>(42)) == 0 * sizeof(int));
static_assert(
	HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x',
                        state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
									 state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 0 * sizeof(int));

// vector_t
static_assert(HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>(),
                                     state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	HasOffsetAlong<vector_t<'x', scalar<int>>, 'x',
                        state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>(),
									 state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 5 * sizeof(int));

static_assert(
	HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
                                     state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(42, 5)) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
                                  state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y',
                       state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(
				  scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
				  state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(42, 42, 5)) ==
              5 * sizeof(int));
static_assert(
	HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
                       state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(
				  scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
				  state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(5, 42, 3)) == 3 * 42 * sizeof(int));

// tuple_t
static_assert(!HasOffsetAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasOffsetAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, lit_t<0>>>>);
static_assert(offset_along<'x'>(pack(scalar<int>(), scalar<int>()) ^ tuple<'x'>(),
									state<state_item<index_in<'x'>, lit_t<0>>>()) == 0 * sizeof(int));
static_assert(HasOffsetAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, lit_t<1>>>>);
static_assert(offset_along<'x'>(pack(scalar<int>(), scalar<int>()) ^ tuple<'x'>(),
									state<state_item<index_in<'x'>, lit_t<1>>>()) == 1 * sizeof(int));
static_assert(!HasOffsetAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, lit_t<2>>>>);

// scalar
static_assert(!HasOffsetAlong<scalar<int>, 'x', state<>>);
static_assert(!HasOffsetAlong<scalar<int>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<scalar<int>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<scalar<int>, 'x',
								  state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(
	!HasOffsetAlong<scalar<int>, 'x',
                        state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

// fix_t
static_assert(!HasOffsetAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(
	!HasOffsetAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<fix_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
                                  state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(HasOffsetAlong<fix_t<'x', vector_t<'y', scalar<int>>, std::size_t>, 'y',
                                 state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'y'>() ^ fix<'x'>(3),
                                     state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 4)) == 4 * sizeof(int));

// set_length_t
static_assert(!HasOffsetAlong<set_length_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(3), state<state_item<index_in<'x'>, std::size_t>>(2)) == 2 * sizeof(int));
static_assert(HasOffsetAlong<set_length_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ set_length<'x'>(6), state<>()) == 0 * sizeof(int));

// rename_t
static_assert(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'x', state<>>);
static_assert(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'y', state<>>);
static_assert(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'y', state<state_item<index_in<'y'>, std::size_t>>>);
static_assert(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(HasOffsetAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ rename<'x', 'y'>(),
									 state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3)) == 3 * sizeof(int));

// join_t
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<index_in<'y'>, std::size_t>>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<index_in<'z'>, std::size_t>>>);
static_assert(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<length_in<'z'>, std::size_t>>>);
static_assert(HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<length_in<'z'>, std::size_t>, state_item<index_in<'z'>, std::size_t>>>);
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ join<'x', 'y', 'z'>(),
									 state<state_item<length_in<'z'>, std::size_t>, state_item<index_in<'z'>, std::size_t>>(5, 3)) == 3 * sizeof(int) + 3 * 5 * sizeof(int));

// shift_t
static_assert(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ shift<'x'>(3), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(6, 2)) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasOffsetAlong<shift_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ shift<'x'>(3), state<>()) == 0 * sizeof(int));

// slice_t
static_assert(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ slice<'x'>(3, 3), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(6, 2)) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasOffsetAlong<slice_t<'x', bcast_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ slice<'x'>(3, 3), state<state_item<index_in<'x'>, std::size_t>>(2)) == 0 * sizeof(int));

// span_t
static_assert(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ span<'x'>(3, 6), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(6, 2)) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasOffsetAlong<span_t<'x', bcast_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ span<'x'>(3, 6), state<state_item<index_in<'x'>, std::size_t>>(2)) == 0 * sizeof(int));

// step_t
static_assert(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'x'>(3, 2), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 4)) == (4 * 2 + 3) * sizeof(int));
static_assert(HasOffsetAlong<step_t<'x', bcast_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ step<'x'>(3, 2), state<state_item<index_in<'x'>, std::size_t>>(4)) == 0 * sizeof(int));

// reverse_t
static_assert(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<>>);
static_assert(HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'x'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 36 * sizeof(int));
static_assert(!HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(HasOffsetAlong<reverse_t<'x', bcast_t<'x', scalar<int>>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ reverse<'x'>(), state<state_item<index_in<'x'>, std::size_t>>(5)) == 0 * sizeof(int));

// into_blocks_t
static_assert(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<>>);
static_assert(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<>>);
static_assert(
	!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
							  state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

							  static_assert(HasOffsetAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 7, 3, 2)) == 3 * sizeof(int));
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 3 * sizeof(int));
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(7, 3, 2)) == 3 * sizeof(int));
static_assert(HasOffsetAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
								 state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 7, 3, 2)) == 2 * 5 * sizeof(int));
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 2 * 5 * sizeof(int));
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(7, 3, 2)) == 2 * 5 * sizeof(int));
static_assert(!HasOffsetAlong<into_blocks_t<'x', 'y', 'z', vector_t<'x', scalar<int>>>, 'x',
								 state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);

// into_blocks_static_t
static_assert(!HasOffsetAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'y', state<>>);
static_assert(!HasOffsetAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'z', state<>>);

static_assert(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'x',
	state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<0>)) == 3 * 16 * sizeof(int));
static_assert(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'y',
	state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<0>)) == 2 * sizeof(int));
static_assert(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'z',
	state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>>);
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<0>)) == 0);

static_assert(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'x',
	state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<1>>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<1>)) == 3 * 16 * sizeof(int));
static_assert(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'y',
	state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<1>>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<1>)) == 2 * sizeof(int));
static_assert(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'z',
	state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<1>>>>);
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<1>)) == 0);

// into_blocks_dynamic_t
static_assert(!HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', scalar<int>>, 'x', state<>>);
static_assert(!HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', scalar<int>>, 'y', state<>>);
static_assert(!HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', scalar<int>>, 'z', state<>>);

static_assert(HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x',
	state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_dynamic<'x', 'x', 'y', 'z'>(),
	state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 3 * 7 * sizeof(int));
static_assert(HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y',
	state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_dynamic<'x', 'x', 'y', 'z'>(),
	state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 2 * sizeof(int));
static_assert(HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z',
	state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_dynamic<'x', 'x', 'y', 'z'>(),
	state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 0);

// merge_blocks_t
static_assert(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'x', state<>>);
static_assert(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'y', state<>>);
static_assert(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'z', state<>>);

static_assert(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'x',
	state<state_item<index_in<'z'>, std::size_t>>>);
static_assert(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'y',
	state<state_item<index_in<'z'>, std::size_t>>>);
static_assert(HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'z',
	state<state_item<index_in<'z'>, std::size_t>>>);
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'x', 'y', 'z'>(),
	state<state_item<index_in<'z'>, std::size_t>>(2)) == 2 * sizeof(int));
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'x', 'y', 'z'>(),
	state<state_item<index_in<'z'>, std::size_t>>(3)) == 3 * sizeof(int));
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'x', 'y', 'z'>(),
	state<state_item<index_in<'z'>, std::size_t>>(5)) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<merge_blocks_t<'y', 'x', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'x',
	state<state_item<index_in<'z'>, std::size_t>>>);
static_assert(!HasOffsetAlong<merge_blocks_t<'y', 'x', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'y',
	state<state_item<index_in<'z'>, std::size_t>>>);
static_assert(HasOffsetAlong<merge_blocks_t<'y', 'x', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'z',
	state<state_item<index_in<'z'>, std::size_t>>>);
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'y', 'x', 'z'>(),
	state<state_item<index_in<'z'>, std::size_t>>(2)) == 2 * 5 * sizeof(int));
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'y', 'x', 'z'>(),
	state<state_item<index_in<'z'>, std::size_t>>(3)) == 1 * sizeof(int));
static_assert(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'y', 'x', 'z'>(),
	state<state_item<index_in<'z'>, std::size_t>>(5)) == (2 * 5 + 1) * sizeof(int));

} // namespace noarr

#endif // NOARR_STRUCTURES_OFFSET_ALONG_HPP
