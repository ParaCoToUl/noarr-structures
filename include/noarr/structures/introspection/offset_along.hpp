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

template<IsDim auto QDim, IsDim auto Dim, class T, class IdxT, IsState State>
struct has_offset_along<QDim, fix_t<Dim, T, IdxT>, State> {
private:
	using Structure = fix_t<Dim, T, IdxT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
		                                                                  structure.sub_state(state), index);
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class LenT, IsState State>
struct has_offset_along<QDim, set_length_t<Dim, T, LenT>, State> {
private:
	using Structure = set_length_t<Dim, T, LenT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
		                                                                  structure.sub_state(state), index);
	}
};




// TODO: implement reorder_t
// TODO: implement hoist_t
// TODO: implement rename_t
// TODO: implement join_t

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, IsState State>
struct has_offset_along<QDim, shift_t<Dim, T, StartT>, State> {
private:
	using Structure = shift_t<Dim, T, StartT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index + structure.start());
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class LenT, IsState State>
struct has_offset_along<QDim, slice_t<Dim, T, StartT, LenT>, State> {
private:
	using Structure = slice_t<Dim, T, StartT, LenT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index + structure.start());
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class EndT, IsState State>
struct has_offset_along<QDim, span_t<Dim, T, StartT, EndT>, State> {
private:
	using Structure = span_t<Dim, T, StartT, EndT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index + structure.start());
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class StrideT, IsState State>
struct has_offset_along<QDim, step_t<Dim, T, StartT, StrideT>, State> {
private:
	using Structure = step_t<Dim, T, StartT, StrideT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:

	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index * structure.stride() + structure.start());
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, reverse_t<Dim, T>, State> {
private:
	using Structure = reverse_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_offset_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto offset(Structure structure, State state, auto index) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), structure.template length<Dim>(state) - make_const<1>() - index);
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                  structure.sub_state(state), index);
		}
	}
};

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

// scalar
static_assert(!HasOffsetAlong<scalar<int>, 'x', state<>>);
static_assert(!HasOffsetAlong<scalar<int>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<scalar<int>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
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
                                 state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(offset_along<'y'>(scalar<int>() ^ vector<'y'>() ^ fix<'x'>(3),
                                     state<state_item<length_in<'y'>, std::size_t>>(5), 4) == 4 * sizeof(int));

// set_length_t
static_assert(!HasOffsetAlong<set_length_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<set_length_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ set_length<'x'>(6), state<>(), 5) == 0 * sizeof(int));
static_assert(HasOffsetAlong<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(3), state<>(), 2) == 2 * sizeof(int));


// TODO: reorder_t
// TODO: hoist_t
// TODO: rename_t
// TODO: join_t

// shift_t
static_assert(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ shift<'x'>(3), state<state_item<length_in<'x'>, std::size_t>>(6), 2) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ shift<'x'>(3), state<state_item<length_in<'x'>, std::size_t>>(6), 2) == 0 * sizeof(int));

// slice_t
static_assert(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ slice<'x'>(3, 3), state<state_item<length_in<'x'>, std::size_t>>(6), 2) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ slice<'x'>(3, 3), state<state_item<length_in<'x'>, std::size_t>>(6), 2) == 0 * sizeof(int));

// span_t
static_assert(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ span<'x'>(3, 6), state<state_item<length_in<'x'>, std::size_t>>(6), 2) == 5 * sizeof(int));
static_assert(!HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ span<'x'>(3, 6), state<state_item<length_in<'x'>, std::size_t>>(6), 2) == 0 * sizeof(int));

// step_t
static_assert(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
static_assert(HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'x'>(3, 2), state<state_item<length_in<'x'>, std::size_t>>(42), 4) == (4 * 2 + 3) * sizeof(int));
static_assert(!HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ step<'x'>(3, 2), state<state_item<length_in<'x'>, std::size_t>>(42), 4) == 0 * sizeof(int));

// reverse_t
static_assert(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(!HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<>>);
static_assert(HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42), 5) == 36 * sizeof(int));
static_assert(!HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(!HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
static_assert(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ reverse<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42), 5) == 0 * sizeof(int));

// TODO: into_blocks_t
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

// TODO: into_blocks_static_t
// TODO: into_blocks_dynamic_t
// TODO: merge_blocks_t

// TODO: zcurve_t

} // namespace noarr

#endif // NOARR_STRUCTURES_OFFSET_ALONG_HPP
