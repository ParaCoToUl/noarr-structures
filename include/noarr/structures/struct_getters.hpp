#ifndef NOARR_STRUCTURES_STRUCT_GETTERS_HPP
#define NOARR_STRUCTURES_STRUCT_GETTERS_HPP

#include <type_traits>

#include "std_ext.hpp"
#include "struct_decls.hpp"
#include "scalar.hpp"
#include "structs.hpp"

namespace noarr {

namespace helpers {
	template<class>
	static constexpr bool wrong_type = false;
	template<char>
	static constexpr bool wrong_dim = false;
}



struct deprecated_tuple_size {
	template<char Dim, class... Ts, class State, std::size_t... Indices>
	static constexpr std::size_t get(const tuple<Dim, Ts...> &tup, State state, std::index_sequence<Indices...>) {
		(void) state; // don't complain about unused parameter in case of empty fold
		return (0 + ... + tup.template sub_structure<Indices>().size(state));
	}
};



template<class Struct>
struct spi_offset;

template<class Struct, class State>
constexpr std::size_t spi_offset_get(Struct str, State state) {
	return spi_offset<Struct>::get(str, state);
}

template<char Dim, std::size_t L, class T>
struct spi_offset<array<Dim, L, T>> {
	template<class State>
	static constexpr std::size_t get(const array<Dim, L, T> &arr, State state) {
		std::size_t index = state.template get<index_in<Dim>>();
		auto sub_structure = arr.sub_structure();
		auto sub_state = state.template remove<length_in<Dim>, index_in<Dim>>();
		std::size_t sub_size = sub_structure.size(sub_state);
		return index*sub_size + spi_offset_get(sub_structure, sub_state);
	}
};

template<char Dim, class T>
struct spi_offset<vector<Dim, T>> {
	template<class State>
	static constexpr std::size_t get(const vector<Dim, T> &vec, State state) {
		std::size_t index = state.template get<index_in<Dim>>();
		auto sub_structure = vec.sub_structure();
		auto sub_state = state.template remove<length_in<Dim>, index_in<Dim>>();
		std::size_t sub_size = sub_structure.size(sub_state);
		return index*sub_size + spi_offset_get(sub_structure, sub_state);
	}
};

template<char Dim, class... Ts>
struct spi_offset<tuple<Dim, Ts...>> {
	template<class State>
	static constexpr std::size_t get(const tuple<Dim, Ts...> &tup, State state) {
		constexpr std::size_t index = state.template get<index_in<Dim>>();
		auto sub_structure = tup.template sub_structure<index>();
		auto sub_state = state.template remove<length_in<Dim>, index_in<Dim>>();
		return deprecated_tuple_size::get(tup, sub_state, std::make_index_sequence<index>()) + spi_offset_get(sub_structure, sub_state);
	}
};

template<class U, class T>
struct spi_offset<setter_t<U, T>> {
	template<class State>
	static constexpr std::size_t get(const setter_t<U, T> &setter, State state) {
		return spi_offset_get(setter.sub_structure(), state.merge(setter.state_update()));
	}
};

template<char Dim, char ViewDim, class T, class ShiftT, class LenT>
struct spi_offset<view_t<Dim, ViewDim, T, ShiftT, LenT>> {
	template<class State>
	static constexpr std::size_t get(const view_t<Dim, ViewDim, T, ShiftT, LenT> &view, State state) {
		auto index = state.template get<index_in<ViewDim>>();
		auto sub_state = state.template remove<index_in<ViewDim>>().template with<index_in<Dim>>(view.shift() + index);
		return spi_offset_get(view.sub_structure(), sub_state);
	}
};

template<class T>
struct spi_offset<scalar<T>> {
	template<class State>
	static constexpr std::size_t get(const scalar<T> &, State) {
		static_assert(State::is_empty, "Unused items in state");
		return 0;
	}
};



template<char QDim, class Struct>
struct spi_length;

template<char QDim, class Struct, class State>
constexpr std::size_t spi_length_get(Struct str, State state) {
	return spi_length<QDim, Struct>::get(str, state);
}

template<char QDim, char Dim, std::size_t L, class T>
struct spi_length<QDim, array<Dim, L, T>> {
	template<class State>
	static constexpr std::size_t get(const array<Dim, L, T> &arr, State state) {
		if constexpr(QDim == Dim) {
			return L;
		} else {
			auto sub_structure = arr.sub_structure();
			auto sub_state = state.template remove<length_in<Dim>>();
			return spi_length_get<QDim>(sub_structure, sub_state);
		}
	}
};

template<char QDim, char Dim, class T>
struct spi_length<QDim, vector<Dim, T>> {
	template<class State>
	static constexpr std::size_t get(const vector<Dim, T> &vec, State state) {
		if constexpr(QDim == Dim) {
			static_assert(helpers::wrong_dim<QDim>, "Length has not been set");
			return 0;
		} else {
			auto sub_structure = vec.sub_structure();
			auto sub_state = state.template remove<length_in<Dim>>();
			return spi_length_get<QDim>(sub_structure, sub_state);
		}
	}
};

template<char QDim, char Dim, class... Ts>
struct spi_length<QDim, tuple<Dim, Ts...>> {
	template<class State>
	static constexpr std::size_t get(const tuple<Dim, Ts...> &tup, State state) {
		if constexpr(QDim == Dim) {
			return sizeof...(Ts);
		} else {
			constexpr std::size_t index = state.template get<index_in<Dim>>();
			auto sub_structure = tup.template sub_structure<index>();
			auto sub_state = state.template remove<length_in<Dim>, index_in<Dim>>();
			return spi_length_get<QDim>(sub_structure, sub_state);
		}
	}
};

template<char QDim, class U, class T>
struct spi_length<QDim, setter_t<U, T>> {
	template<class State>
	static constexpr std::size_t get(const setter_t<U, T> &setter, State state) {
		auto sub_state = state.merge(setter.state_update());
		static_assert(!decltype(sub_state)::template contains<index_in<QDim>>, "Index in this dimension is overriden by a substructure");
		if constexpr(decltype(sub_state)::template contains<length_in<QDim>>) {
			return sub_state.template get<length_in<QDim>>();
		} else {
			return spi_length_get<QDim>(setter.sub_structure(), sub_state);
		}
	}
};

template<char QDim, char Dim, char ViewDim, class T, class ShiftT, class LenT>
struct spi_length<QDim, view_t<Dim, ViewDim, T, ShiftT, LenT>> {
	template<class State>
	static constexpr std::size_t get(const view_t<Dim, ViewDim, T, ShiftT, LenT> &view, State state) {
		if constexpr(QDim == ViewDim && !std::is_same_v<LenT, view_default_len>) {
			return view.len() - view.shift();
		} else {
			if constexpr(State::template contains<index_in<ViewDim>>) {
				auto index = state.template get<index_in<ViewDim>>();
				auto sub_state = state.template remove<index_in<ViewDim>>().template with<index_in<Dim>>(view.shift() + index);
				if constexpr(QDim == ViewDim)
					return spi_length_get<Dim>(view.sub_structure(), sub_state) - view.shift();
				else
					return spi_length_get<QDim>(view.sub_structure(), sub_state);
			} else {
				if constexpr(QDim == ViewDim)
					return spi_length_get<Dim>(view.sub_structure(), state) - view.shift();
				else
					return spi_length_get<QDim>(view.sub_structure(), state);
			}
		}
	}
};

template<char QDim, class T>
struct spi_length<QDim, scalar<T>> {
	template<class State>
	static constexpr std::size_t get(const scalar<T> &, State) {
		static_assert(helpers::wrong_dim<QDim>, "Index in this dimension is not accepted by any substructure");
		return 0;
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_GETTERS_HPP
