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
	constexpr std::tuple<TS...> sub_structures() const noexcept { return sub_structures(std::index_sequence_for<TS...>()); }
	using description = struct_description<
		char_pack<'t', 'u', 'p', 'l', 'e'>,
		dims_impl<Dim>,
		dims_impl<>,
		structure_param<TS>...>;

	constexpr tuple() noexcept = default;
	constexpr tuple(TS... ss) noexcept : base(ss...) {}

	static_assert(!(TS::struct_type::template any_accept<Dim> || ...), "Dimension name already used");
	using struct_type = dep_function_type<Dim, typename TS::struct_type...>;

private:
	template<std::size_t... IS>
	constexpr std::tuple<TS...> sub_structures(std::index_sequence<IS...>) const noexcept {
		return std::tuple(sub_structure<IS>()...);
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
};

template<char Dim>
struct vector<Dim> {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return vector<Dim, Struct>(s); }
};

template<class Struct, class ProtoStruct, class = std::enable_if_t<ProtoStruct::is_proto_struct, decltype(std::declval<Struct>().sub_structures())>>
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
