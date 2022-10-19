#ifndef NOARR_STRUCTURES_STRUCT_DECLS_HPP
#define NOARR_STRUCTURES_STRUCT_DECLS_HPP

#include <tuple>

#include "utility.hpp"
#include "signature.hpp"

namespace noarr {

/**
 * @brief a pack of the template parameters of the structure
 * 
 * @tparam Params: template parameters of the structure
 */
template<class... Params>
struct struct_params;

template<class>
struct structure_param;

template<class>
struct type_param;

template<class T, T V>
struct value_param;

template<char Dim>
struct dim_param;

template<class StructInner, class StructOuter, class State>
constexpr std::size_t offset_of(StructOuter structure, State state) noexcept {
	if constexpr(std::is_same_v<StructInner, StructOuter>) {
		// TODO check that state only contains relevant lengths
		return 0;
	} else {
		return structure.template strict_offset_of<StructInner>(state);
	}
}

template<class StructInner, class StructOuter, class State>
constexpr auto state_at(StructOuter structure, State state) noexcept {
	if constexpr(std::is_same_v<StructInner, StructOuter>) {
		return state;
	} else {
		return structure.template strict_state_at<StructInner>(state);
	}
}

namespace helpers {

template<class T, class = void>
struct is_struct_impl : std::false_type {};
template<class T>
struct is_struct_impl<T, std::void_t<typename T::signature>> : std::true_type {
	static_assert(is_signature<typename T::signature>());
};

} // namespace helpers

/**
 * @brief returns whether the type `T` meets the criteria for structures
 * 
 * @tparam T: the input type
 */
template<class T>
using is_struct = helpers::is_struct_impl<T>;

template<class Struct, class ProtoStruct, class = std::enable_if_t<is_struct<Struct>::value && ProtoStruct::is_proto_struct>>
constexpr auto operator ^(Struct s, ProtoStruct p) {
	return p.instantiate_and_construct(s);
}

struct neutral_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return s; }
};

template<class InnerProtoStruct, class OuterProtoStruct, class = std::enable_if_t<InnerProtoStruct::is_proto_struct && OuterProtoStruct::is_proto_struct>>
struct compose_proto : contain<InnerProtoStruct, OuterProtoStruct> {
	using base = contain<InnerProtoStruct, OuterProtoStruct>;
	using base::base;
	constexpr compose_proto(InnerProtoStruct i, OuterProtoStruct o) noexcept : base(i, o) {}

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return base::template get<1>().instantiate_and_construct(base::template get<0>().instantiate_and_construct(s));
	}
};

template<class InnerProtoStruct, class OuterProtoStruct>
constexpr compose_proto<InnerProtoStruct, OuterProtoStruct> operator ^(InnerProtoStruct i, OuterProtoStruct o) {
	return compose_proto(i, o);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_DECLS_HPP
