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

template<class... Args>
struct pack : contain<Args...> {
	using contain<Args...>::contain;

	explicit constexpr pack(Args... args) noexcept : contain<Args...>(args...) {}
};

// this is for the consistency of packs of packs
template<class... Args>
pack(pack<Args...>) -> pack<pack<Args...>>;

template<class ProtoStruct>
struct to_each : ProtoStruct {
	using ProtoStruct::ProtoStruct;

	to_each(ProtoStruct p) noexcept : ProtoStruct(p) {}
};

template<class ProtoStruct>
to_each(ProtoStruct) -> to_each<ProtoStruct>;

namespace helpers {

template<class T>
struct is_struct_impl : std::false_type {};
template<class T> requires (is_signature_v<typename T::signature>)
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

template<class... Structs, class ProtoStruct, std::size_t... Indices> requires (is_proto_struct_impl<ProtoStruct>::value)
constexpr auto pass_pack(pack<Structs...> s, ProtoStruct p, std::index_sequence<Indices...>) noexcept {
	return p.instantiate_and_construct(s.template get<Indices>()...);
}

template<class Arg, class... Args, std::size_t... Indices>
constexpr auto pass_pack(Arg s, pack<Args...> p, std::index_sequence<Indices...>) noexcept {
	return pack(s ^ p.template get<Indices>()...);
}

template<class... Structs, class Arg, std::size_t... Indices>
constexpr auto pass_pack(pack<Structs...> s, to_each<Arg> p, std::index_sequence<Indices...>) noexcept {
	return pack(s.template get<Indices>() ^ p...);
}

} // namespace helpers

/**
 * @brief returns whether the type `T` meets the criteria for structures
 * 
 * @tparam T: the input type
 */
template<class T>
using is_struct = helpers::is_struct_impl<T>;

template<class T>
static constexpr auto is_struct_v = is_struct<T>::value;

/**
 * @brief returns whether the type `T` meets the criteria for proto-structures
 * 
 * @tparam T: the input type
 */
template<class T>
using is_proto_struct = helpers::is_proto_struct_impl<T>;

template<class T>
static constexpr auto is_proto_struct_v = is_proto_struct<T>::value;

template<bool PreservesLayout = false, class F>
constexpr auto make_proto(F f) noexcept {
	return helpers::make_proto_impl<F, PreservesLayout>(f);
}

template<class Struct, class ProtoStruct> requires (is_struct_v<Struct> && is_proto_struct_v<ProtoStruct>)
constexpr auto operator ^(Struct s, ProtoStruct p) noexcept {
	return p.instantiate_and_construct(s);
}

template<class... Structs, class ProtoStruct> requires (is_proto_struct_v<ProtoStruct> && ... && is_struct_v<Structs>)
constexpr auto operator ^(pack<Structs...> s, ProtoStruct p) noexcept {
	return helpers::pass_pack(s, p, std::make_index_sequence<sizeof...(Structs)>());
}

template<class Arg, class... Args>
constexpr auto operator ^(Arg s, pack<Args...> p) noexcept {
	return helpers::pass_pack(s, p, std::make_index_sequence<sizeof...(Args)>());
}

template<class... Structs, class Arg> requires (... && is_struct_v<Structs>)
constexpr auto operator ^(pack<Structs...> s, to_each<Arg> p) noexcept {
	return helpers::pass_pack(s, p, std::make_index_sequence<sizeof...(Structs)>());
}

struct neutral_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return s; }
};

template<class InnerProtoStructPack, class OuterProtoStruct>
struct compose_proto;

template<class... InnerProtoStructs, class OuterProtoStruct>
struct compose_proto<pack<InnerProtoStructs...>, OuterProtoStruct> : contain<pack<InnerProtoStructs...>, OuterProtoStruct> {
	using base = contain<pack<InnerProtoStructs...>, OuterProtoStruct>;
	using base::base;

	explicit constexpr compose_proto(pack<InnerProtoStructs...> i, OuterProtoStruct o) noexcept : base(i, o) {}

	static constexpr bool proto_preserves_layout = (OuterProtoStruct::proto_preserves_layout && ... && InnerProtoStructs::proto_preserves_layout);

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return s ^ base::template get<0>() ^ base::template get<1>();
	}

	template<class... Structs> requires (sizeof...(Structs) != 1)
	constexpr auto instantiate_and_construct(Structs... s) const noexcept {
		return pack(s...) ^ base::template get<0>() ^ base::template get<1>();
	}
};

template<class InnerProtoStruct, class OuterProtoStruct> requires (is_proto_struct_v<InnerProtoStruct> && is_proto_struct_v<OuterProtoStruct>)
constexpr compose_proto<pack<InnerProtoStruct>, OuterProtoStruct> operator ^(InnerProtoStruct i, OuterProtoStruct o) noexcept {
	return compose_proto<pack<InnerProtoStruct>, OuterProtoStruct>(pack(i), o);
}

template<class... InnerProtoStructs, class OuterProtoStruct> requires (is_proto_struct_v<OuterProtoStruct> && ... && is_proto_struct_v<InnerProtoStructs>)
constexpr compose_proto<pack<InnerProtoStructs...>, OuterProtoStruct> operator ^(pack<InnerProtoStructs...> i, OuterProtoStruct o) noexcept {
	return compose_proto<pack<InnerProtoStructs...>, OuterProtoStruct>(i, o);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCTS_COMMON_HPP
