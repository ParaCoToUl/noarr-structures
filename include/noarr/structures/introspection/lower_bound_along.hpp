#ifndef NOARR_STRUCTURES_LOWER_BOUND_ALONG_HPP
#define NOARR_STRUCTURES_LOWER_BOUND_ALONG_HPP

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

// TODO: could be more elegant to define min_state and end_state instead of min and end

namespace noarr {

namespace helpers {

// implicitly implements specialization for scalar and all incorrect structures
template<IsDim auto QDim, class T, IsState State>
struct has_lower_bound_along : std::false_type {};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_lower_bound_along<QDim, bcast_t<Dim, T>, State> {
private:
	using Structure = bcast_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {

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
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			return true;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
		}
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return make_const<0>(); // lower bound of a broadcasted dimension is always 0
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return make_const<0>(); // lower bound of a broadcasted dimension is always 0
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return make_const<0>(); // the canonical offset of the lower bound of a broadcasted dimension is always the
			                        // lowest index
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return min; // the canonical offset of the lower bound of a broadcasted dimension is always the lowest index
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class... Ts, IsState State>
struct has_lower_bound_along<QDim, tuple_t<Dim, Ts...>, State> {
private:
	using Structure = tuple_t<Dim, Ts...>;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (State::template contains<index_in<Dim>>) {
			using index_t = state_get_t<State, index_in<Dim>>;

			if constexpr (requires {
							  index_t::value;
							  requires (index_t::value < sizeof...(Ts));
						  }) {
				constexpr std::size_t index = state_get_t<State, index_in<Dim>>::value;

				if constexpr (QDim == Dim) {
					return false;
				} else {
					using sub_structure_t = typename Structure::template sub_structure_t<index>;

					return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
				}
			} else {
				return false;
			}
		} else if constexpr (QDim == Dim) {
			return true;
		} else {
			return false;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		if constexpr (State::template contains<index_in<Dim>>) {
			// QDim != Dim
			using index_t = state_get_t<State, index_in<Dim>>;
			constexpr std::size_t index = index_t::value;

			using sub_structure_t = typename Structure::template sub_structure_t<index>;

			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
		} else {
			// QDim == Dim
			return true;
		}
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (State::template contains<index_in<Dim>>) {
			// QDim != Dim
			using index_t = state_get_t<State, index_in<Dim>>;
			constexpr std::size_t index = index_t::value;

			using sub_structure_t = typename Structure::template sub_structure_t<index>;

			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(index), structure.sub_state(state));

		} else {
			using namespace constexpr_arithmetic;
			// QDim == Dim
			return make_const<0>();
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value && requires {
		decltype(min)::value;
		requires ((decltype(min)::value) < sizeof...(Ts));
		decltype(end)::value;
		requires ((decltype(min)::value) < decltype(end)::value);
	}
	{
		if constexpr (State::template contains<index_in<Dim>>) {
			// QDim != Dim
			using index_t = state_get_t<State, index_in<Dim>>;
			constexpr std::size_t index = index_t::value;

			using sub_structure_t = typename Structure::template sub_structure_t<index>;

			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(index), structure.sub_state(state), min, end);

		} else {
			// QDim == Dim
			constexpr std::size_t index = decltype(min)::value;

			using sub_structure_t = typename Structure::template sub_structure_t<index>;

			return offset_of<sub_structure_t>(structure, state.template with<index_in<Dim>>(min));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (State::template contains<index_in<Dim>>) {
			// QDim != Dim
			using index_t = state_get_t<State, index_in<Dim>>;

			constexpr std::size_t index = index_t::value;

			using sub_structure_t = typename Structure::template sub_structure_t<index>;

			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(index), structure.sub_state(state));

		} else {
			using namespace constexpr_arithmetic;
			// QDim == Dim
			return make_const<0>();
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value && requires {
		decltype(min)::value;
		requires ((decltype(min)::value) < sizeof...(Ts));
		decltype(end)::value;
		requires ((decltype(min)::value) < decltype(end)::value);
	}
	{
		if constexpr (State::template contains<index_in<Dim>>) {
			// QDim != Dim
			using index_t = state_get_t<State, index_in<Dim>>;

			constexpr std::size_t index = index_t::value;

			using sub_structure_t = typename Structure::template sub_structure_t<index>;

			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(index), structure.sub_state(state), min, end);

		} else {
			// QDim == Dim
			return min;
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_lower_bound_along<QDim, vector_t<Dim, T>, State> {
private:
	using Structure = vector_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
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
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			return true;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
		}
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return make_const<0>(); // lower bound of a vectorized dimension is always 0
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return structure.sub_structure().size(structure.sub_state(state)) * min;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return make_const<0>(); // the canonical offset of the lower bound of a vectorized dimension is always the
			                        // lowest index
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return min; // the canonical offset of the lower bound of a vectorized dimension is always the lowest index
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class IdxT, IsState State>
struct has_lower_bound_along<QDim, fix_t<Dim, T, IdxT>, State> {
private:
	using Structure = fix_t<Dim, T, IdxT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
		                                                                              structure.sub_state(state));
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
			structure.sub_structure(), structure.sub_state(state), min, end);
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(structure.sub_structure(),
		                                                                                 structure.sub_state(state));
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
			structure.sub_structure(), structure.sub_state(state), min, end);
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class LenT, IsState State>
struct has_lower_bound_along<QDim, set_length_t<Dim, T, LenT>, State> {
private:
	using Structure = set_length_t<Dim, T, LenT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
		                                                                              structure.sub_state(state));
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
			structure.sub_structure(), structure.sub_state(state), min, end);
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(structure.sub_structure(),
		                                                                                 structure.sub_state(state));
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
			structure.sub_structure(), structure.sub_state(state), min, end);
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_lower_bound_along<QDim, hoist_t<Dim, T>, State> {
private:
	using Structure = hoist_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
		                                                                              structure.sub_state(state));
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
			structure.sub_structure(), structure.sub_state(state), min, end);
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(structure.sub_structure(),
		                                                                                 structure.sub_state(state));
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
			structure.sub_structure(), structure.sub_state(state), min, end);
	}
};

template<IsDim auto QDim, class T, auto... DimPairs, IsState State>
requires IsDimPack<decltype(DimPairs)...> && (sizeof...(DimPairs) % 2 == 0)
struct has_lower_bound_along<QDim, rename_t<T, DimPairs...>, State> {
private:
	using Structure = rename_t<T, DimPairs...>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	constexpr static auto QDimNew =
		helpers::rename_dim<QDim, typename Structure::external, typename Structure::internal>::dim;

	static constexpr bool get_value() noexcept {
		if constexpr (Structure::internal::template contains<QDim> && !Structure::external::template contains<QDim>) {
			return false;
		} else {
			return has_lower_bound_along<QDimNew, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		return has_lower_bound_along<QDimNew, sub_structure_t, sub_state_t>::is_monotonic();
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDimNew, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
		                                                                                 structure.sub_state(state));
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		return has_lower_bound_along<QDimNew, sub_structure_t, sub_state_t>::lower_bound(
			structure.sub_structure(), structure.sub_state(state), min, end);
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDimNew, sub_structure_t, sub_state_t>::lower_bound_at(structure.sub_structure(),
		                                                                                    structure.sub_state(state));
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		return has_lower_bound_along<QDimNew, sub_structure_t, sub_state_t>::lower_bound_at(
			structure.sub_structure(), structure.sub_state(state), min, end);
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, IsState State>
struct has_lower_bound_along<QDim, shift_t<Dim, T, StartT>, State> {
private:
	using Structure = shift_t<Dim, T, StartT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				sub_structure, sub_state, start, sub_structure.template length<Dim>(sub_state));
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min + start, end + start);
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					   sub_structure, sub_state, start, sub_structure.template length<Dim>(sub_state) - start) -
			       start;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					   structure.sub_structure(), structure.sub_state(state), min + start, end + start) -
			       start;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class LenT, IsState State>
struct has_lower_bound_along<QDim, slice_t<Dim, T, StartT, LenT>, State> {
private:
	using Structure = slice_t<Dim, T, StartT, LenT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				sub_structure, sub_state, start, start + structure.len());
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min + start, end + start);
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					   sub_structure, sub_state, start, start + structure.len()) -
			       start;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					   structure.sub_structure(), structure.sub_state(state), min + start, end + start) -
			       start;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class EndT, IsState State>
struct has_lower_bound_along<QDim, span_t<Dim, T, StartT, EndT>, State> {
private:
	using Structure = span_t<Dim, T, StartT, EndT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr auto is_monotonic() noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);
			const auto start = structure.start();
			const auto end = structure.end();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(sub_structure, sub_state,
			                                                                             start, end);
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min + start, end + start);
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);
			const auto start = structure.start();
			const auto end = structure.end();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(sub_structure, sub_state,
			                                                                                start, end) -
			       start;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			using namespace constexpr_arithmetic;
			const auto start = structure.start();
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					   structure.sub_structure(), structure.sub_state(state), min + start, end + start) -
			       start;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_lower_bound_along<QDim, reverse_t<Dim, T>, State> {
private:
	using Structure = reverse_t<Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
		                                                                              structure.sub_state(state));
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		const auto sub_structure = structure.sub_structure();
		const auto sub_state = structure.sub_state(state);

		using namespace constexpr_arithmetic;

		if constexpr (QDim == Dim) {
			const auto sub_length = sub_structure.template length<Dim>(sub_state);
			const auto sub_last = sub_length - make_const<1>();
			const auto in_min = sub_length - end;
			const auto in_max = sub_last - min;

			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(sub_structure, sub_state,
			                                                                             in_min, in_max);
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(sub_structure, sub_state, min,
			                                                                              end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		const auto sub_structure = structure.sub_structure();
		const auto sub_state = structure.sub_state(state);

		using namespace constexpr_arithmetic;

		if constexpr (QDim == Dim) {
			return sub_structure.template length<Dim>(sub_state) - make_const<1>() -
			       has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(sub_structure, sub_state);
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(sub_structure, sub_state);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		const auto sub_structure = structure.sub_structure();
		const auto sub_state = structure.sub_state(state);

		using namespace constexpr_arithmetic;

		if constexpr (QDim == Dim) {
			const auto sub_length = sub_structure.template length<Dim>(sub_state);
			const auto sub_last = sub_length - make_const<1>();
			const auto in_min = sub_length - end;
			const auto in_max = sub_last - min;

			return sub_last - has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
								  sub_structure, sub_state, in_min, in_max);
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(sub_structure, sub_state,
			                                                                                 min, end);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, class T, IsState State>
requires (DimMajor != DimMinor)
struct has_lower_bound_along<QDim, into_blocks_t<Dim, DimMajor, DimMinor, T>, State> {
private:
	using Structure = into_blocks_t<Dim, DimMajor, DimMinor, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMinor) {
			// the lower bound of the DimMinor dimension is always equal to the lower bound of the Dim dimension
			if constexpr (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::value) {
				return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
			} else {
				return false;
			}
		} else if constexpr (QDim == DimMajor) {
			if constexpr (Structure::template has_length<DimMinor, State>()) {
				// the lower bound of the DimMajor dimension is always equal to the lower bound of the Dim dimension
				if constexpr (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::value) {
					return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else if constexpr (QDim == Dim) {
			// the Dim dimension is consumed by into_blocks
			return false;
		} else {
			// propagate to the substructure
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
		} else if constexpr (QDim == DimMajor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
		}
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), make_const<0>(),
				structure.template length<DimMinor>(state));
		} else if constexpr (QDim == DimMajor) {
			return make_const<0>();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		} else if constexpr (QDim == DimMajor) {
			const auto len = structure.template length<DimMinor>(state);
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min * len, end * len);
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), make_const<0>(),
				structure.template length<DimMinor>(state));
		} else if constexpr (QDim == DimMajor) {
			if (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					structure.sub_structure(), structure.sub_state(state)) == make_const<0>()) {
				return make_const<0>();
			} else {
				return structure.template length<DimMinor>(state) - make_const<1>();
			}
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		} else if constexpr (QDim == DimMajor) {
			if (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					structure.sub_structure(), structure.sub_state(state)) == make_const<0>()) {
				return min;
			} else {
				return end - make_const<1>();
			}
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class T,
         class MinorLenT, IsState State>
requires (DimIsBorder != DimMajor) && (DimIsBorder != DimMinor) && (DimMajor != DimMinor)
struct has_lower_bound_along<QDim, into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, T, MinorLenT>, State> {
private:
	using Structure = into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, T, MinorLenT>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMinor) {
			if constexpr (!State::template contains<index_in<DimIsBorder>>) {
				return false;
			} else if constexpr (Structure::template has_length<DimMinor, State>()) {
				if constexpr (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::value) {
					return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else if constexpr (QDim == DimMajor) {
			if constexpr (!State::template contains<index_in<DimIsBorder>>) {
				return false;
			} else if constexpr (Structure::template has_length<DimMinor, State>()) {
				if constexpr (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::value) {
					return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else if constexpr (QDim == DimIsBorder) {
			if constexpr (State::template contains<index_in<DimIsBorder>>) {
				return false;
			} else if constexpr (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::value) {
				return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
			} else {
				return false;
			}
		} else if constexpr (QDim == Dim) {
			return false;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
		} else if constexpr (QDim == DimMajor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
		} else if constexpr (QDim == DimIsBorder) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
		}
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			const auto len = structure.template length<DimMinor>(state);
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), make_const<0>(), len);
		} else if constexpr (QDim == DimMajor || QDim == DimIsBorder) {
			return make_const<0>();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		} else if constexpr (QDim == DimMajor) {
			const auto len = structure.template length<DimMinor>(state);
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min * len, end * len);
		} else if constexpr (QDim == DimIsBorder) {
			return make_const<0>();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), make_const<0>(),
				structure.template length<DimMinor>(state));
		} else if constexpr (QDim == DimMajor) {
			if (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					structure.sub_structure(), structure.sub_state(state)) == make_const<0>()) {
				return make_const<0>();
			} else {
				return structure.template length<DimMinor>(state) - make_const<1>();
			}
		} else if constexpr (QDim == DimIsBorder) {
			if (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					structure.sub_structure(), structure.sub_state(state)) == make_const<0>()) {
				return make_const<0>();
			} else {
				return make_const<1>();
			}
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		} else if constexpr (QDim == DimMajor) {
			const auto len = structure.template length<DimMinor>(state);
			if (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					structure.sub_structure(), structure.sub_state(state)) == make_const<0>()) {
				return min;
			} else {
				return end - make_const<1>();
			}
		} else if constexpr (QDim == DimIsBorder) {
			const auto len = structure.template length<DimMinor>(state);
			if (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					structure.sub_structure(), structure.sub_state(state)) == make_const<0>()) {
				return min;
			} else {
				return end - make_const<1>();
			}
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent, class T,
         IsState State>
requires (DimMajor != DimMinor) && (DimMinor != DimIsPresent) && (DimIsPresent != DimMajor)
struct has_lower_bound_along<QDim, into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, T>, State> {
private:
	using Structure = into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == DimMinor) {
			if constexpr (Structure::template has_length<DimMinor, State>()) {
				if constexpr (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::value) {
					return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else if constexpr (QDim == DimMajor) {
			if constexpr (Structure::template has_length<DimMinor, State>()) {
				if constexpr (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::value) {
					return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else if constexpr (QDim == DimIsPresent) {
			if constexpr (Structure::template has_length<DimMinor, State>()) {
				if constexpr (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::value) {
					return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else if constexpr (QDim == Dim) {
			return false;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
		} else if constexpr (QDim == DimMajor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
		} else if constexpr (QDim == DimIsPresent) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::is_monotonic();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
		}
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			const auto len = structure.template length<DimMinor>(state);
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), make_const<0>(), len);
		} else if constexpr (QDim == DimMajor || QDim == DimIsPresent) {
			return make_const<0>();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		} else if constexpr (QDim == DimMajor) {
			const auto len = structure.template length<DimMinor>(state);
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min * len, end * len);
		} else if constexpr (QDim == DimIsPresent) {
			return make_const<0>();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimMinor) {
			return has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), make_const<0>(),
				structure.template length<DimMinor>(state));
		} else if constexpr (QDim == DimMajor) {
			if (has_lower_bound_along<Dim, sub_structure_t, sub_state_t>::lower_bound_at(
					structure.sub_structure(), structure.sub_state(state)) == make_const<0>()) {
				return make_const<0>();
			} else {
				return structure.template length<DimMinor>(state) - make_const<1>();
			}
		} else if constexpr (QDim == DimIsPresent) {
			return make_const<0>();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim, class T, IsState State>
struct has_lower_bound_along<QDim, merge_blocks_t<DimMajor, DimMinor, Dim, T>, State> {
private:
	using Structure = merge_blocks_t<DimMajor, DimMinor, Dim, T>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr bool get_value() noexcept {
		if constexpr (QDim == Dim) {
			if constexpr (has_lower_bound_along<DimMajor, sub_structure_t, sub_state_t>::value &&
			              has_lower_bound_along<DimMinor, sub_structure_t, sub_state_t>::value) {
				return has_lower_bound_along<DimMajor, sub_structure_t, sub_state_t>::is_monotonic() &&
				       has_lower_bound_along<DimMinor, sub_structure_t, sub_state_t>::is_monotonic();
			} else {
				return false;
			}
		} else if constexpr (QDim == DimMajor || QDim == DimMinor) {
			return false;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::value;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();

	static constexpr bool is_monotonic() noexcept
	requires value
	{
		if constexpr (QDim == Dim) {
			return has_lower_bound_along<DimMajor, sub_structure_t, sub_state_t>::is_monotonic() &&
			       has_lower_bound_along<DimMinor, sub_structure_t, sub_state_t>::is_monotonic();
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::is_monotonic();
		}
	}

	static constexpr auto lower_bound(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);

			const auto major =
				has_lower_bound_along<DimMajor, sub_structure_t, sub_state_t>::lower_bound(sub_structure, sub_state);
			const auto minor =
				has_lower_bound_along<DimMinor, sub_structure_t, sub_state_t>::lower_bound(sub_structure, sub_state);
			return major + minor;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(structure.sub_structure(),
			                                                                              structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);

			const auto length = sub_structure.template length<DimMinor>(sub_state);
			const auto major = has_lower_bound_along<DimMajor, sub_structure_t, sub_state_t>::lower_bound(
				sub_structure, sub_state, min / length, end / length);
			const auto minor = has_lower_bound_along<DimMinor, sub_structure_t, sub_state_t>::lower_bound(
				sub_structure, sub_state, min % length, end % length);
			return major + minor;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);

			const auto major =
				has_lower_bound_along<DimMajor, sub_structure_t, sub_state_t>::lower_bound_at(sub_structure, sub_state);
			const auto minor =
				has_lower_bound_along<DimMinor, sub_structure_t, sub_state_t>::lower_bound_at(sub_structure, sub_state);
			return major + minor;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state));
		}
	}

	static constexpr auto lower_bound_at(Structure structure, State state, auto min, auto end) noexcept
	requires value
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			const auto sub_structure = structure.sub_structure();
			const auto sub_state = structure.sub_state(state);

			const auto length = sub_structure.template length<DimMinor>(sub_state);
			const auto major = has_lower_bound_along<DimMajor, sub_structure_t, sub_state_t>::lower_bound_at(
				sub_structure, sub_state, min / length, end / length);
			const auto minor = has_lower_bound_along<DimMinor, sub_structure_t, sub_state_t>::lower_bound_at(
				sub_structure, sub_state, min % length, end % length);
			return major + minor;
		} else {
			return has_lower_bound_along<QDim, sub_structure_t, sub_state_t>::lower_bound_at(
				structure.sub_structure(), structure.sub_state(state), min, end);
		}
	}
};

} // namespace helpers

template<class T, auto Dim, class State>
concept HasLowerBoundAlong = requires {
	requires IsStruct<T>;
	requires IsState<State>;
	requires IsDim<decltype(Dim)>;

	requires helpers::has_lower_bound_along<Dim, T, State>::value;
};

template<auto Dim, class T, class State>
constexpr auto lower_bound_along(T structure, State state) noexcept
requires HasLowerBoundAlong<T, Dim, State>
{
	return helpers::has_lower_bound_along<Dim, T, State>::lower_bound(structure, state);
}

template<auto Dim, class T, class State>
constexpr auto lower_bound_at(T structure, State state) noexcept
requires HasLowerBoundAlong<T, Dim, State>
{
	return helpers::has_lower_bound_along<Dim, T, State>::lower_bound_at(structure, state);
}

template<auto Dim, class T, class State>
constexpr auto lower_bound_along(T structure, State state, auto min, auto end) noexcept
requires HasLowerBoundAlong<T, Dim, State>
{
	return helpers::has_lower_bound_along<Dim, T, State>::lower_bound(structure, state, min, end);
}

template<auto Dim, class T, class State>
constexpr auto lower_bound_at(T structure, State state, auto min, auto end) noexcept
requires HasLowerBoundAlong<T, Dim, State>
{
	return helpers::has_lower_bound_along<Dim, T, State>::lower_bound_at(structure, state, min, end);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_LOWER_BOUND_ALONG_HPP
