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

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

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
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

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

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

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
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

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

// TODO: implement tuple_t

template<IsDim auto QDim, IsDim auto Dim, class T, class IdxT, IsState State>
struct has_stride_along<QDim, fix_t<Dim, T, IdxT>, State> {
private:
	using Structure = fix_t<Dim, T, IdxT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class LenT, IsState State>
struct has_stride_along<QDim, set_length_t<Dim, T, LenT>, State> {
private:
	using Structure = set_length_t<Dim, T, LenT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
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

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, IsState State>
struct has_stride_along<QDim, shift_t<Dim, T, StartT>, State> {
private:
	using Structure = shift_t<Dim, T, StartT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class LenT, IsState State>
struct has_stride_along<QDim, slice_t<Dim, T, StartT, LenT>, State> {
private:
	using Structure = slice_t<Dim, T, StartT, LenT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class EndT, IsState State>
struct has_stride_along<QDim, span_t<Dim, T, StartT, EndT>, State> {
private:
	using Structure = span_t<Dim, T, StartT, EndT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state));
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class StrideT, IsState State>
struct has_stride_along<QDim, step_t<Dim, T, StartT, StrideT>, State> {
private:
	using Structure = step_t<Dim, T, StartT, StrideT>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		using namespace constexpr_arithmetic;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
		                                                                    structure.sub_state(state)) * structure.stride();
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_stride_along<QDim, reverse_t<Dim, T>, State> {
private:
	using Structure = reverse_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		return has_stride_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto stride(Structure structure, State state) noexcept
	requires value
	{
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		using namespace constexpr_arithmetic;

		using sub_stride_t = decltype(has_stride_along<QDim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
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

	static constexpr bool get_value() noexcept {
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

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
		using sub_structure_t = typename Structure::sub_structure_t;
		using sub_state_t = typename Structure::template sub_state_t<State>;

		if constexpr (QDim == DimMinor) {
			return has_stride_along<Dim, sub_structure_t, sub_state_t>::stride(structure.sub_structure(),
			                                                                   structure.sub_state(state));
		} else if constexpr (QDim == DimMajor) {
			return structure.template length<DimMinor>(state) *
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

// TODO: implement into_blocks_static_t
// TODO: implement into_blocks_dynamic_t
// TODO: implement merge_blocks_t

// TODO: implement zcurve_t

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

// TODO: tuple_t

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

// TODO: implement reorder_t
// TODO: implement hoist_t
// TODO: implement rename_t
// TODO: implement join_t

// TODO: implement shift_t
// TODO: implement slice_t
// TODO: implement span_t
// TODO: implement step_t
// TODO: implement reverse_t

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

// TODO: implement into_blocks_static_t
// TODO: implement into_blocks_dynamic_t
// TODO: implement merge_blocks_t

// TODO: implement zcurve_t

} // namespace noarr

#endif // NOARR_STRUCTURES_STRIDE_ALONG_HPP
