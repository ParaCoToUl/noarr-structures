#ifndef NOARR_STRUCTURES_OFFSET_ALONG_HPP
#define NOARR_STRUCTURES_OFFSET_ALONG_HPP

#include <cstddef>

#include <type_traits>

#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

#include "../structs/bcast.hpp"
#include "../structs/blocks.hpp"
#include "../structs/layouts.hpp"
#include "../structs/setters.hpp"
#include "../structs/slice.hpp"
#include "../structs/views.hpp"

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

	static constexpr bool get_value() noexcept { return has_offset_along<QDim, sub_structure_t, sub_state_t>::value; }

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
		if constexpr (state_contains<State, index_in<Dim>>) {
			using index_t = state_get_t<State, index_in<Dim>>;

			if constexpr (requires {
							  index_t::value;
							  requires (index_t::value < sizeof...(Ts));
						  }) {
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
struct has_offset_along<QDim, fix_t<Dim, T, IdxT>, State> : generic_has_offset_along<QDim, fix_t<Dim, T, IdxT>, State> {
};

template<IsDim auto QDim, IsDim auto Dim, class T, class LenT, IsState State>
struct has_offset_along<QDim, set_length_t<Dim, T, LenT>, State>
	: generic_has_offset_along<QDim, set_length_t<Dim, T, LenT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, IsState State>
struct has_offset_along<QDim, hoist_t<Dim, T>, State> : generic_has_offset_along<QDim, hoist_t<Dim, T>, State> {};

template<IsDim auto QDim, class T, auto... DimPairs, IsState State>
requires IsDimPack<decltype(DimPairs)...> && (sizeof...(DimPairs) % 2 == 0)
struct has_offset_along<QDim, rename_t<T, DimPairs...>, State> {
private:
	using Structure = rename_t<T, DimPairs...>;
	using sub_structure_t = typename Structure::sub_structure_t;
	using sub_state_t = typename Structure::template sub_state_t<State>;

	static constexpr auto QDimNew =
		helpers::rename_dim<QDim, typename Structure::external, typename Structure::internal>::dim;

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
			return has_offset_along<DimA, sub_structure_t, sub_state_t>::value &&
			       has_offset_along<DimB, sub_structure_t, sub_state_t>::value;
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
			                                                                    structure.sub_state(state)) +
			       has_offset_along<DimB, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, IsState State>
struct has_offset_along<QDim, shift_t<Dim, T, StartT>, State>
	: generic_has_offset_along<QDim, shift_t<Dim, T, StartT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class LenT, IsState State>
struct has_offset_along<QDim, slice_t<Dim, T, StartT, LenT>, State>
	: generic_has_offset_along<QDim, slice_t<Dim, T, StartT, LenT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class EndT, IsState State>
struct has_offset_along<QDim, span_t<Dim, T, StartT, EndT>, State>
	: generic_has_offset_along<QDim, span_t<Dim, T, StartT, EndT>, State> {};

template<IsDim auto QDim, IsDim auto Dim, class T, class StartT, class StrideT, IsState State>
struct has_offset_along<QDim, step_t<Dim, T, StartT, StrideT>, State>
	: generic_has_offset_along<QDim, step_t<Dim, T, StartT, StrideT>, State> {};

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
			return offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMinor>>(make_const<0>()))) -
			       offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(
										 make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimMinor) {
			return offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>>(make_const<0>()))) -
			       offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(
										 make_const<0>(), make_const<0>())));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class T,
         class MinorLenT, IsState State>
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
			return offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMinor>>(make_const<0>()))) -
			       offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(
										 make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimMinor) {
			return offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>>(make_const<0>()))) -
			       offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(
										 make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimIsBorder) {
			return offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(
										 make_const<0>(), make_const<0>()))) -
			       offset_along<Dim>(
					   structure.sub_structure(),
					   structure.sub_state(
						   state.template with<index_in<DimMajor>, index_in<DimMinor>, index_in<DimIsBorder>>(
							   make_const<0>(), make_const<0>(), make_const<0>())));
		} else {
			return has_offset_along<QDim, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
			                                                                    structure.sub_state(state));
		}
	}
};

template<IsDim auto QDim, IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent, class T,
         IsState State>
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
			return offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMinor>>(make_const<0>()))) -
			       offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(
										 make_const<0>(), make_const<0>())));
		} else if constexpr (QDim == DimMinor) {
			return offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>>(make_const<0>()))) -
			       offset_along<Dim>(structure.sub_structure(),
			                         structure.sub_state(state.template with<index_in<DimMajor>, index_in<DimMinor>>(
										 make_const<0>(), make_const<0>())));
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
			return has_offset_along<DimMajor, sub_structure_t, sub_state_t>::value &&
			       has_offset_along<DimMinor, sub_structure_t, sub_state_t>::value;
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
			                                                                        structure.sub_state(state)) +
			       has_offset_along<DimMinor, sub_structure_t, sub_state_t>::offset(structure.sub_structure(),
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

} // namespace noarr

#endif // NOARR_STRUCTURES_OFFSET_ALONG_HPP
