#ifndef NOARR_STRUCTURES_UNIFORM_ALONG_HPP
#define NOARR_STRUCTURES_UNIFORM_ALONG_HPP

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
struct is_uniform_along : std::false_type {};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct is_uniform_along<QDim, bcast_t<Dim, T>, State> {
private:
	using Structure = bcast_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			return State::template contains<length_in<Dim>> && !State::template contains<index_in<Dim>>;
		} else {
			return is_uniform_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct is_uniform_along<QDim, vector_t<Dim, T>, State> {
private:
	using Structure = vector_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == Dim) {
			return State::template contains<length_in<Dim>> && !State::template contains<index_in<Dim>>;
		} else {
			return is_uniform_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

// TODO: implement tuple_t

template<IsDim auto QDim, IsDim auto Dim, class T, class IdxT, IsState State>
struct is_uniform_along<QDim, fix_t<Dim, T, IdxT>, State> {
private:
	using Structure = fix_t<Dim, T, IdxT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return is_uniform_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto QDim, IsDim auto Dim, class T, class LenT, IsState State>
struct is_uniform_along<QDim, set_length_t<Dim, T, LenT>, State> {
private:
	using Structure = set_length_t<Dim, T, LenT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return is_uniform_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

// TODO: implement reorder_t
// TODO: implement hoist_t
// TODO: implement rename_t
// TODO: implement join_t

// TODO: implement shift_t
// TODO: implement slice_t
// TODO: implement span_t
// TODO: implement step_t
// TODO: implement reverse_t

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, class T, IsState State>
requires (DimMajor != DimMinor)
struct is_uniform_along<QDim, into_blocks_t<Dim, DimMajor, DimMinor, T>, State> {
private:
	using Structure = into_blocks_t<Dim, DimMajor, DimMinor, T>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == DimMinor) {
			return is_uniform_along<Dim, sub_structure_t, sub_state_t>::value;
		} else if constexpr (QDim == DimMajor) {
			if constexpr (Structure::template has_length<DimMinor, State>()) {
				return is_uniform_along<Dim, sub_structure_t, sub_state_t>::value;
			} else {
				return false;
			}
		} else if constexpr (QDim == Dim) {
			return false;
		} else {
			return is_uniform_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

// TODO: implement into_blocks_static_t
// TODO: implement into_blocks_dynamic_t
// TODO: implement merge_blocks_t

// TODO: implement zcurve_t

} // namespace helpers

template<class T, auto Dim, class State>
concept IsUniformAlong = requires {
	requires IsStruct<T>;
	requires IsState<State>;
	requires IsDim<decltype(Dim)>;

	requires helpers::is_uniform_along<Dim, T, State>::value;
};

// tests

// bcast_t
static_assert(IsUniformAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!IsUniformAlong<bcast_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!IsUniformAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!IsUniformAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!IsUniformAlong<bcast_t<'x', scalar<int>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

// vector_t
static_assert(IsUniformAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!IsUniformAlong<vector_t<'x', scalar<int>>, 'x', state<>>);
static_assert(!IsUniformAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!IsUniformAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!IsUniformAlong<vector_t<'x', scalar<int>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

static_assert(
	IsUniformAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	!IsUniformAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(IsUniformAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y',
                             state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(IsUniformAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
                             state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);

// TODO: tuple_t

// scalar
static_assert(!IsUniformAlong<scalar<int>, 'x', state<>>);
static_assert(!IsUniformAlong<scalar<int>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!IsUniformAlong<scalar<int>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
static_assert(!IsUniformAlong<scalar<int>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

// fix_t
static_assert(!IsUniformAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(
	!IsUniformAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!IsUniformAlong<fix_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(IsUniformAlong<fix_t<'x', vector_t<'y', scalar<int>>, std::size_t>, 'y',
                             state<state_item<length_in<'y'>, std::size_t>>>);

// set_length_t
static_assert(!IsUniformAlong<set_length_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
static_assert(IsUniformAlong<set_length_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
static_assert(IsUniformAlong<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);

// into_blocks_t
static_assert(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<>>);
static_assert(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<>>);
static_assert(
	!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(
	!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
static_assert(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y',
                              state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);

static_assert(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x', state<>>);
static_assert(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y', state<>>);
static_assert(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
                              state<state_item<length_in<'x'>, std::size_t>>>);
static_assert(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
                              state<state_item<length_in<'y'>, std::size_t>>>);
static_assert(IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
                             state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
static_assert(IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
                             state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);

} // namespace noarr

#endif // NOARR_STRUCTURES_UNIFORM_ALONG_HPP
