#ifndef NOARR_STRUCTURES_TO_STRUCT_HPP
#define NOARR_STRUCTURES_TO_STRUCT_HPP

#include <type_traits>

#include "../base/structs_common.hpp"

namespace noarr {

template<class T>
struct to_struct : std::false_type {};

template<class T>
constexpr bool to_struct_v = to_struct<T>::value;

template<class T>
using to_struct_t = typename to_struct<T>::type;

template<class T>
concept ToStruct = to_struct_v<std::remove_cvref_t<T>>;

template<ToStruct T>
constexpr decltype(auto) convert_to_struct(T &&t) noexcept {
	using type = std::remove_cvref_t<T>;
	return to_struct<type>::convert(std::forward<T>(t));
}

template<IsStruct T>
struct to_struct<T> : std::true_type {
	using type = std::remove_cvref_t<T>;

	[[nodiscard]]
	static constexpr decltype(auto) convert(T t) noexcept {
		return t;
	}
};

template<IsStruct T>
struct to_struct<pack<T>> : std::true_type {
	using type = T;

	static constexpr decltype(auto) convert(const pack<T> &p) noexcept { return p.template get<0>(); }
};

} // namespace noarr

#endif // NOARR_STRUCTURES_TO_STRUCT_HPP
