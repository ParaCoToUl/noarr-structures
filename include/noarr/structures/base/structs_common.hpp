#ifndef NOARR_STRUCTURES_STRUCTS_COMMON_HPP
#define NOARR_STRUCTURES_STRUCTS_COMMON_HPP

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
constexpr auto offset_of(StructOuter structure, State state) noexcept {
	if constexpr(std::is_same_v<StructInner, StructOuter>) {
		return constexpr_arithmetic::make_const<0>();
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

template<class T>
struct is_struct_impl : std::false_type {};
template<class T> requires is_signature<typename T::signature>::value
struct is_struct_impl<T> : std::true_type {
	static_assert(is_signature<typename T::signature>());
};

template<class T>
struct is_proto_struct_impl : std::false_type {};
template<class T> requires std::is_same_v<decltype(T::proto_preserves_layout), const bool>
struct is_proto_struct_impl<T> : std::true_type {};

template<class F, bool PreservesLayout>
struct make_proto_impl {
	static constexpr bool proto_preserves_layout = PreservesLayout;

	F f_;

	explicit constexpr make_proto_impl(F f) noexcept : f_(f) {}


	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return f_(s); }
};
} // namespace helpers

/**
 * @brief returns whether the type `T` meets the criteria for structures
 * 
 * @tparam T: the input type
 */
template<class T>
using is_struct = helpers::is_struct_impl<T>;

template<class T>
using is_proto_struct = helpers::is_proto_struct_impl<T>;


template<bool PreservesLayout = false, class F>
constexpr auto make_proto(F f) noexcept {
	return helpers::make_proto_impl<F, PreservesLayout>(f);
}

template<class Struct, class ProtoStruct> requires (is_struct<Struct>::value && is_proto_struct<ProtoStruct>::value)
constexpr auto operator ^(Struct s, ProtoStruct p) {
	return p.instantiate_and_construct(s);
}

struct neutral_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return s; }
};

template<class InnerProtoStruct, class OuterProtoStruct, bool PreservesLayout = InnerProtoStruct::proto_preserves_layout && OuterProtoStruct::proto_preserves_layout>
struct compose_proto : contain<InnerProtoStruct, OuterProtoStruct> {
	using base = contain<InnerProtoStruct, OuterProtoStruct>;
	using base::base;
	constexpr compose_proto(InnerProtoStruct i, OuterProtoStruct o) noexcept : base(i, o) {}

	static constexpr bool proto_preserves_layout = PreservesLayout;

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

#endif // NOARR_STRUCTURES_STRUCTS_COMMON_HPP
