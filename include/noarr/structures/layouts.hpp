#ifndef NOARR_STRUCTURES_LAYOUTS_HPP
#define NOARR_STRUCTURES_LAYOUTS_HPP

#include "structs_common.hpp"
#include "contain.hpp"
#include "scalar.hpp"
#include "state.hpp"
#include "signature.hpp"

namespace noarr {

/**
 * @brief tuple
 * 
 * @tparam Dim dimmension added by the structure
 * @tparam T,TS... substructure types
 */
template<char Dim, class... TS>
struct tuple : contain<TS...> {
	using base = contain<TS...>;

	template<std::size_t Index>
	constexpr auto sub_structure() const noexcept { return base::template get<Index>(); }
	static constexpr char name[] = "tuple";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<TS>...>;

	constexpr tuple() noexcept = default;
	constexpr tuple(TS... ss) noexcept : base(ss...) {}

	static_assert(!(TS::signature::template any_accept<Dim> || ...), "Dimension name already used");
	using signature = dep_function_sig<Dim, typename TS::signature...>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set tuple length");
		return size_inner(is, state.template remove<index_in<Dim>>());
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set tuple length");
		static_assert(State::template contains<index_in<Dim>>, "All indices must be set");
		static_assert(state_get_t<State, index_in<Dim>>::value || true, "Tuple index must be set statically, wrap it in idx<> (e.g. replace 42 with idx<42>)");
		constexpr std::size_t index = state_get_t<State, index_in<Dim>>::value;
		auto sub_state = state.template remove<index_in<Dim>>(); // TODO remove all indices for size_inner
		return size_inner(std::make_index_sequence<index>(), sub_state) + offset_of<Sub>(sub_structure<index>(), sub_state);
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set tuple length");
		if constexpr(QDim == Dim) {
			static_assert(!State::template contains<index_in<Dim>>, "Index already set");
			// TODO check remaining state
			return sizeof...(TS);
		} else {
			static_assert(State::template contains<index_in<Dim>>, "Tuple indices must be set");
			constexpr std::size_t index = state.template get<index_in<Dim>>();
			return sub_structure<index>().template length<QDim>(state.template remove<index_in<Dim>>());
		}
	}

	template<class Sub, class State>
	constexpr void strict_state_at(State) const noexcept {
		static_assert(value_always_false<Dim>, "A tuple cannot be used in this context");
	}

private:
	static constexpr std::index_sequence_for<TS...> is = {};

	template<std::size_t... IS, class State>
	constexpr std::size_t size_inner(std::index_sequence<IS...>, State sub_state) const noexcept {
		(void) sub_state; // don't complain about unused parameter in case of empty fold
		return (0 + ... + sub_structure<IS>().size(sub_state));
	}
};

/**
 * @brief a structure representing an array with a dynamicly specifiable index (all indices point to the same substructure, with a different offset)
 * 
 * @tparam Dim: the dimmension name added by the array
 * @tparam T: the type of the substructure the array contains
 */
template<char Dim, std::size_t L, class T = void>
struct array : contain<T> {
	static constexpr char name[] = "array";
	using params = struct_params<
		dim_param<Dim>,
		value_param<std::size_t, L>,
		structure_param<T>>;

	constexpr array() noexcept = default;
	explicit constexpr array(T sub_structure) noexcept : contain<T>(sub_structure) {}

	constexpr T sub_structure() const noexcept { return contain<T>::template get<0>(); }

	static_assert(!T::signature::template any_accept<Dim>, "Dimension name already used");
	using signature = function_sig<Dim, static_arg_length<L>, typename T::signature>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set array length");
		return L * sub_structure().size(state.template remove<index_in<Dim>>());
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set array length");
		static_assert(State::template contains<index_in<Dim>>, "All indices must be set");
		std::size_t index = state.template get<index_in<Dim>>();
		auto sub_struct = sub_structure();
		auto sub_state = state.template remove<index_in<Dim>>();
		return index * sub_struct.size(sub_state) + offset_of<Sub>(sub_struct, sub_state);
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set array length");
		if constexpr(QDim == Dim) {
			static_assert(!State::template contains<index_in<Dim>>, "Index already set");
			// TODO check remaining state
			return L;
		} else {
			return sub_structure().template length<QDim>(state.template remove<index_in<Dim>>());
		}
	}

	template<class Sub, class State>
	constexpr void strict_state_at(State) const noexcept {
		static_assert(value_always_false<Dim>, "An array cannot be used in this context");
	}
};

template<char Dim, std::size_t L>
struct array<Dim, L> {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return array<Dim, L, Struct>(s); }
};

/**
 * @brief unsized vector ready to be resized to the desired size, this vector does not have size yet
 * 
 * @tparam Dim: the dimmension name added by the vector
 * @tparam T: type of the substructure the vector contains
 */
template<char Dim, class T = void>
struct vector : contain<T> {
	static constexpr char name[] = "vector";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>>;

	constexpr vector() noexcept = default;
	explicit constexpr vector(T sub_structure) noexcept : contain<T>(sub_structure) {}

	constexpr T sub_structure() const noexcept { return contain<T>::template get<0>(); }

	static_assert(!T::signature::template any_accept<Dim>, "Dimension name already used");
	using signature = function_sig<Dim, unknown_arg_length, typename T::signature>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		static_assert(State::template contains<length_in<Dim>>, "Unknown vector length");
		std::size_t len = state.template get<length_in<Dim>>();
		return len * sub_structure().size(state.template remove<index_in<Dim>, length_in<Dim>>());
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		static_assert(State::template contains<index_in<Dim>>, "All indices must be set");
		std::size_t index = state.template get<index_in<Dim>>();
		auto sub_struct = sub_structure();
		auto sub_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		return index * sub_struct.size(sub_state) + offset_of<Sub>(sub_struct, sub_state);
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		if constexpr(QDim == Dim) {
			static_assert(!State::template contains<index_in<Dim>>, "Index already set");
			static_assert(State::template contains<length_in<Dim>>, "This length has not been set yet");
			// TODO check remaining state
			return state.template get<length_in<Dim>>();
		} else {
			return sub_structure().template length<QDim>(state.template remove<index_in<Dim>, length_in<Dim>>());
		}
	}

	template<class Sub, class State>
	constexpr void strict_state_at(State) const noexcept {
		static_assert(value_always_false<Dim>, "A vector cannot be used in this context");
	}
};

template<char Dim>
struct vector<Dim> {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return vector<Dim, Struct>(s); }
};

} // namespace noarr

#endif // NOARR_STRUCTURES_LAYOUTS_HPP
