#ifndef NOARR_STRUCTURES_STRUCTS_HPP
#define NOARR_STRUCTURES_STRUCTS_HPP

#include "struct_decls.hpp"
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
	constexpr std::tuple<TS...> sub_structures() const noexcept { return sub_structures(is); }
	using description = struct_description<
		char_pack<'t', 'u', 'p', 'l', 'e'>,
		dims_impl<Dim>,
		dims_impl<>,
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
		static_assert(state_get_t<State, index_in<Dim>>::value || true, "Tuple index must be set statically, add _idx to the index (e.g. replace 42 with 42_idx)");
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
		static_assert(always_false_dim<Dim>, "A tuple cannot be used in this context");
	}

private:
	static constexpr std::index_sequence_for<TS...> is = {};

	template<std::size_t... IS>
	constexpr std::tuple<TS...> sub_structures(std::index_sequence<IS...>) const noexcept {
		return std::tuple(sub_structure<IS>()...);
	}

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
	constexpr std::tuple<T> sub_structures() const noexcept { return std::tuple<T>(contain<T>::template get<0>()); }
	using description = struct_description<
		char_pack<'a', 'r', 'r', 'a', 'y'>,
		dims_impl<Dim>,
		dims_impl<>,
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
		static_assert(always_false_dim<Dim>, "An array cannot be used in this context");
	}
};

template<char Dim, std::size_t L>
struct array<Dim, L> {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return array<Dim, L, Struct>(s); }
};

/**
 * @brief unsized vector ready to be resized to the desired size, this vector does not have size yet
 * 
 * @tparam Dim: the dimmension name added by the vector
 * @tparam T: type of the substructure the vector contains
 */
template<char Dim, class T = void>
struct vector : contain<T> {
	constexpr std::tuple<T> sub_structures() const noexcept { return std::tuple<T>(contain<T>::template get<0>()); }
	using description = struct_description<
		char_pack<'v', 'e', 'c', 't', 'o', 'r'>,
		dims_impl<Dim>,
		dims_impl<>,
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
		static_assert(always_false_dim<Dim>, "A vector cannot be used in this context");
	}
};

template<char Dim>
struct vector<Dim> {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return vector<Dim, Struct>(s); }
};

template<class Struct, class ProtoStruct, class = std::enable_if_t<is_struct<Struct>::value && ProtoStruct::is_proto_struct>>
constexpr auto operator ^(Struct s, ProtoStruct p) {
	return p.instantiate_and_construct(s);
}

static constexpr struct unit_struct_t {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return s; }
} unit_struct;

namespace helpers {

template<class InnerProtoStruct, class OuterProtoStruct, class = std::enable_if_t<InnerProtoStruct::is_proto_struct && OuterProtoStruct::is_proto_struct>>
struct gcompose : contain<InnerProtoStruct, OuterProtoStruct> {
	using base = contain<InnerProtoStruct, OuterProtoStruct>;
	using base::base;
	explicit constexpr gcompose(InnerProtoStruct i, OuterProtoStruct o) noexcept : base(i, o) {}

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept {
		return base::template get<1>().instantiate_and_construct(base::template get<0>().instantiate_and_construct(s));
	}
};

}

template<class InnerProtoStruct, class OuterProtoStruct>
constexpr helpers::gcompose<InnerProtoStruct, OuterProtoStruct> operator ^(InnerProtoStruct i, OuterProtoStruct o) {
	return helpers::gcompose(i, o);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCTS_HPP
