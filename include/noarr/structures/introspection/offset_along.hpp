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

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, bcast_t<Dim, T>, State> {
private:
	using Structure = bcast_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			if constexpr (State::template contains<length_in<Dim>> && !State::template contains<index_in<Dim>>) {
				using namespace constexpr_arithmetic;
				// TODO: check if this is correct
				constexpr auto zero = make_const<0>();
				return has_offset_of<sub_structure_t, Structure,
				                     decltype(std::declval<State>().template with<index_in<Dim>>(zero))>();
			} else {
				return false;
			}
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return make_const<0>(); // offset of a broadcasted dimension is always 0
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, vector_t<Dim, T>, State> {
private:
	using Structure = vector_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			if constexpr (State::template contains<length_in<Dim>> && !State::template contains<index_in<Dim>>) {
				using namespace constexpr_arithmetic;
				constexpr auto zero = make_const<0>();
				return has_offset_of<sub_structure_t, Structure,
				                     decltype(std::declval<State>().template with<index_in<Dim>>(zero))>();
			} else {
				return false;
			}
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return index * structure.sub_structure().size(structure.sub_state(state));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index);
		}
	}
};

// TODO: implement tuple_t

// TODO: implement fix_t
// TODO: implement set_length_t


// TODO: implement reorder_t
// TODO: implement hoist_t
// TODO: implement rename_t
// TODO: implement join_t

// TODO: implement shift_t
// TODO: implement slice_t
// TODO: implement span_t
// TODO: implement step_t
// TODO: implement reverse_t

// TODO: implement into_blocks_t
// TODO: implement into_blocks_static_t
// TODO: implement into_blocks_dynamic_t
// TODO: implement merge_blocks_t

// TODO: implement zcurve_t

} // namespace helpers

template<class T, auto Dim, class State>
concept HasOffsetAlong = requires {
	requires IsStruct<T>;
	requires IsState<State>;
	requires IsDim<decltype(Dim)>;

	requires helpers::has_offset_along<Dim, T, State>::value;
};

template<auto Dim, class T, class State>
constexpr auto offset_along(T structure, State state, auto index) noexcept
requires HasOffsetAlong<T, Dim, State>
{
	return helpers::has_offset_along<Dim, T, State>::offset(structure, state, index);
}

// tests

// bcast_t
static_assert(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
                                     state<state_item<length_in<'x'>, std::size_t>>(42), 5) == 0);
static_assert(!HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	!HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x',
                        state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

// vector_t
static_assert(HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>(),
                                     state<state_item<length_in<'x'>, std::size_t>>(42), 5) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x',
                        state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(
	HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
                                     state<state_item<length_in<'y'>, std::size_t>>(42), 5) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
                                  state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y',
                       state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(
				  scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
				  state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>(42, 42), 5) ==
              5 * sizeof(int));
static_assert(
	HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
                       state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(offset_along<'x'>(
				  scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
				  state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>(5, 42), 3) == 3 * 42 * sizeof(int));

// TODO: tuple_t

// TODO: fix_t
// TODO: set_length_t


// TODO: reorder_t
// TODO: hoist_t
// TODO: rename_t
// TODO: join_t

// TODO: shift_t
// TODO: slice_t
// TODO: span_t
// TODO: step_t
// TODO: reverse_t

// TODO: into_blocks_t
// TODO: into_blocks_static_t
// TODO: into_blocks_dynamic_t
// TODO: merge_blocks_t

// TODO: zcurve_t

} // namespace noarr

#endif // NOARR_STRUCTURES_OFFSET_ALONG_HPP
