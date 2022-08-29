#ifndef NOARR_STRUCTURES_STRUCTS_HPP
#define NOARR_STRUCTURES_STRUCTS_HPP

#include "struct_decls.hpp"
#include "contain.hpp"
#include "scalar.hpp"
#include "state.hpp"
#include "type.hpp"

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

	static_assert(!(TS::struct_type::template any_accept<Dim> || ...), "Dimension name already used");
	using struct_type = dep_function_type<Dim, typename TS::struct_type...>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set tuple length");
		return size_inner(is, state.template remove<index_in<Dim>>());
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set tuple length");
		static_assert(State::template contains<index_in<Dim>>, "All indices must be set");
		static_assert(State::template get_t<index_in<Dim>>::value || true, "Tuple index must be set statically, add _idx to the index (e.g. replace 42 with 42_idx)");
		constexpr std::size_t index = State::template get_t<index_in<Dim>>::value;
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

	static_assert(!T::struct_type::template any_accept<Dim>, "Dimension name already used");
	using struct_type = function_type<Dim, static_arg_length<L>, typename T::struct_type>;

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

	static_assert(!T::struct_type::template any_accept<Dim>, "Dimension name already used");
	using struct_type = function_type<Dim, unknown_arg_length, typename T::struct_type>;

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

struct view_default_len {};

// TODO replace with shift, slice, rename
template<char Dim, char ViewDim, class T, class ShiftT, class LenT>
struct view_t : contain<T, ShiftT, LenT> {
	using base = contain<T, ShiftT, LenT>;

	constexpr std::tuple<T> sub_structures() const noexcept { return std::tuple<T>(base::template get<0>()); }
	using description = struct_description<
		char_pack<'v', 'i', 'e', 'w'>,
		dims_impl<ViewDim>,
		dims_impl<Dim>,
		structure_param<T>>;

	constexpr view_t(T sub_structure, ShiftT shift, LenT len) noexcept : base(sub_structure, shift, len) {}

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }
	constexpr ShiftT shift() const noexcept { return base::template get<1>(); }
	constexpr LenT len() const noexcept { return base::template get<2>(); }

	static_assert(T::struct_type::template all_accept<Dim>, "The structure does not have a dimension of this name");
	static_assert(ViewDim == Dim || !T::struct_type::template any_accept<ViewDim>, "Dimension of this name already exists");
private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetType>
	struct dim_replacement<function_type<Dim, ArgLength, RetType>> { using type = function_type<ViewDim, dynamic_arg_length/*TODO arg_length_from_t<LenT>*/, RetType>; };
	template<class... RetTypes>
	struct dim_replacement<dep_function_type<Dim, RetTypes...>> {
		using original = dep_function_type<Dim, RetTypes...>;
		static_assert(ShiftT::value || true, "Cannot view a tuple dimension as dynamic");
		static_assert(LenT::value || true, "Cannot view a tuple dimension as dynamic");
		static constexpr std::size_t shift = ShiftT::value;
		static constexpr std::size_t len = LenT::value;

		template<class Indices = std::make_index_sequence<len>>
		struct pack_helper;
		template<std::size_t... Indices>
		struct pack_helper<std::index_sequence<Indices...>> { using type = dep_function_type<ViewDim, typename original::ret_type<Indices-shift>...>; };

		using type = typename pack_helper<>::type;
	};
public:
	using struct_type = typename T::struct_type::replace<dim_replacement, Dim>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		// TODO check and translate
		return sub_structure().size(state);
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		// TODO length
		static_assert(State::template contains<index_in<ViewDim>>, "All indices must be set");
		std::size_t index = state.template get<index_in<ViewDim>>();
		auto sub_state = state.template remove<index_in<ViewDim>>().template with<index_in<Dim>>(shift() + index);
		return offset_of<Sub>(sub_structure(), sub_state);
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		// TODO clean up the code after split
		if constexpr(QDim == ViewDim && !std::is_same_v<LenT, view_default_len>) {
			// TODO check remaining state
			return len() - shift();
		} else {
			if constexpr(State::template contains<index_in<ViewDim>>) {
				auto index = state.template get<index_in<ViewDim>>();
				auto sub_state = state.template remove<index_in<ViewDim>>().template with<index_in<Dim>>(shift() + index);
				if constexpr(QDim == ViewDim)
					return sub_structure().template length<Dim>(sub_state) - shift();
				else
					return sub_structure().template length<QDim>(sub_state);
			} else {
				if constexpr(QDim == ViewDim)
					return sub_structure().template length<Dim>(state) - shift();
				else
					return sub_structure().template length<QDim>(state);
			}
		}
	}
};

template<char Dim, char ViewDim, class ShiftT, class LenT>
struct view_proto : contain<ShiftT, LenT> {
	using base = contain<ShiftT, LenT>;
	using base::base;

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return view_t<Dim, ViewDim, Struct, ShiftT, LenT>(s, base::template get<0>(), base::template get<1>()); }
};

template<char Dim>
auto view() { return view_proto<Dim, Dim, std::integral_constant<std::size_t, 0>, view_default_len>(); }

template<char Dim, class ShiftT>
auto view(const ShiftT &shift) { return view_proto<Dim, Dim, ShiftT, view_default_len>(shift); }

template<char Dim, class ShiftT, class LenT>
auto view(const ShiftT &shift, const LenT &len) { return view_proto<Dim, Dim, ShiftT, LenT>(shift, len); }

template<char Dim, char ViewDim>
auto view() { return view_proto<Dim, Dim, std::integral_constant<std::size_t, 0>, view_default_len>(); }

template<char Dim, char ViewDim, class ShiftT>
auto view(const ShiftT &shift) { return view_proto<Dim, Dim, ShiftT, view_default_len>(shift); }

template<char Dim, char ViewDim, class ShiftT, class LenT>
auto view(const ShiftT &shift, const LenT &len) { return view_proto<Dim, Dim, ShiftT, LenT>(shift, len); }

static constexpr struct unit_struct_t {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return s; }
} unit_struct;

namespace helpers {

template<class InnerProtoStruct, class OuterProtoStruct, class = std::enable_if_t<InnerProtoStruct::is_proto_struct && OuterProtoStruct::is_proto_struct>>
struct gcompose : contain<InnerProtoStruct, OuterProtoStruct> {
	using base = contain<InnerProtoStruct, OuterProtoStruct>;
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
