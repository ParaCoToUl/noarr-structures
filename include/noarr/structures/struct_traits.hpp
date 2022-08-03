#ifndef NOARR_STRUCTURES_STRUCT_TRAITS_HPP
#define NOARR_STRUCTURES_STRUCT_TRAITS_HPP

#include <type_traits>

#include "std_ext.hpp"
#include "struct_decls.hpp"
#include "scalar.hpp"
#include "structs.hpp"

namespace noarr {

template<bool IsCube, bool IsPoint>
struct struct_traits {
	static constexpr bool is_cube = IsCube;
	static constexpr bool is_point = IsPoint;
	static constexpr struct_traits<false, IsPoint> non_cube = {};
	static constexpr struct_traits<IsCube, false> non_point = {};
};



template<class Struct>
struct spi_traits;

template<class Struct, class State>
constexpr auto spi_traits_get(Struct str, State state) {
	std::abort();
	return spi_traits<Struct>::get(str, state);
}

template<class Struct, class State>
using spi_traits_t = decltype(spi_traits_get(std::declval<Struct>(), std::declval<State>()));

template<char Dim, std::size_t L, class T>
struct spi_traits<array<Dim, L, T>> {
	template<class State>
	static auto get(array<Dim, L, T> arr, State state) {
		auto sub_traits = spi_traits_get(arr.sub_structure(), state.template remove<index_in<Dim>, length_in<Dim>>());
		if constexpr(State::template contains<index_in<Dim>>)
			return sub_traits;
		else
			return sub_traits.non_point;
	}
};

template<char Dim, class T>
struct spi_traits<vector<Dim, T>> {
	template<class State>
	static auto get(vector<Dim, T> vec, State state) {
		if constexpr(State::template contains<length_in<Dim>>) {
			auto sub_traits = spi_traits_get(vec.sub_structure(), state.template remove<index_in<Dim>, length_in<Dim>>());
			if constexpr(State::template contains<index_in<Dim>>)
				return sub_traits;
			else
				return sub_traits.non_point;
		} else {
			return struct_traits<false, false>();
		}
	}
};

template<char Dim, class... Ts>
struct spi_traits<tuple<Dim, Ts...>> {
	template<class State>
	static auto get(const tuple<Dim, Ts...> &tup, State state) {
		if constexpr(State::template contains<index_in<Dim>>) {
			static constexpr std::size_t index = state.template get<index_in<Dim>>();
			auto sub_structure = tup.template sub_structure<index>();
			auto sub_state = state.template remove<index_in<Dim>>();
			return spi_traits_get(sub_structure, sub_state);
		} else {
			return struct_traits<false, false>();
		}
	}
};

template<class U, class T>
struct spi_traits<setter_t<U, T>> {
	template<class State>
	static auto get(setter_t<U, T> setter, State state) {
		return spi_traits_get(setter.sub_structure(), state.merge(setter.state_update()));
	}
};

template<char Dim, char ViewDim, class T, class ShiftT, class LenT>
struct spi_traits<view_t<Dim, ViewDim, T, ShiftT, LenT>> {
	template<class State>
	static auto get(const view_t<Dim, ViewDim, T, ShiftT, LenT> &view, State state) {
		if constexpr(State::template contains<index_in<ViewDim>>) {
			auto index = state.template get<index_in<ViewDim>>();
			auto sub_state = state.template remove<index_in<ViewDim>>().template with<index_in<Dim>>(view.shift() + index);
			return spi_traits_get(view.sub_structure(), sub_state);
		} else {
			return spi_traits_get(view.sub_structure(), state);
		}
	}
};

template<class T>
struct spi_traits<scalar<T>> {
	template<class State>
	static struct_traits<true, true> get(scalar<T>, State) {
		std::abort();
	}
};



template<class Struct>
struct spi_type;

template<class Struct, class State>
constexpr auto spi_type_get(Struct str, State state) {
	std::abort();
	return spi_type<Struct>::get(str, state);
}

template<class Struct, class State>
using spi_type_t = typename decltype(spi_type_get(std::declval<Struct>(), std::declval<State>()))::type;

template<char Dim, std::size_t L, class T>
struct spi_type<array<Dim, L, T>> {
	template<class State>
	static auto get(array<Dim, L, T> arr, State state) {
		return spi_type_get(arr.sub_structure(), state);
	}
};

template<char Dim, class T>
struct spi_type<vector<Dim, T>> {
	template<class State>
	static auto get(vector<Dim, T> vec, State state) {
		return spi_type_get(vec.sub_structure(), state);
	}
};

template<char Dim, class... Ts>
struct spi_type<tuple<Dim, Ts...>> {
	template<class State>
	static auto get(const tuple<Dim, Ts...> &tup, State state) {
		static constexpr std::size_t index = state.template get<index_in<Dim>>();
		auto sub_structure = tup.template sub_structure<index>();
		auto sub_state = state.template remove<index_in<Dim>>();
		return spi_type_get(sub_structure, sub_state);
	}
};

template<class U, class T>
struct spi_type<setter_t<U, T>> {
	template<class State>
	static auto get(setter_t<U, T> setter, State state) {
		return spi_type_get(setter.sub_structure(), state.merge(setter.state_update()));
	}
};

template<char Dim, char ViewDim, class T, class ShiftT, class LenT>
struct spi_type<view_t<Dim, ViewDim, T, ShiftT, LenT>> {
	template<class State>
	static auto get(const view_t<Dim, ViewDim, T, ShiftT, LenT> &view, State state) {
		auto index = state.template get<index_in<ViewDim>>();
		auto sub_state = state.template remove<index_in<ViewDim>>().template with<index_in<Dim>>(view.shift() + index);
		return spi_type_get(view.sub_structure(), state.merge(sub_state));
	}
};

template<class T>
struct spi_type<scalar<T>> {
	template<class State>
	static spi_type<scalar<T>> get(scalar<T>, State) {
		std::abort();
	}

	using type = T;
};



/**
 * @brief returns whether the structure is a point (a structure with no dimensions, or with all dimensions being fixed)
 * 
 * @tparam T: the structure
 */
template<class T>
struct is_point : std::integral_constant<bool, spi_traits_t<T, state<>>::is_point> {};

/**
 * @brief returns whether a structure is a cube (its dimension and dimension of its substructures, recursively, are all dynamic)
 * 
 * @tparam T: the structure
 */
template<class T>
struct is_cube : std::integral_constant<bool, spi_traits_t<T, state<>>::is_cube> {};

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_TRAITS_HPP
