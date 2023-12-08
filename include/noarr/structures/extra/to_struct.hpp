#ifndef NOARR_STRUCTURES_TO_STRUCT_HPP
#define NOARR_STRUCTURES_TO_STRUCT_HPP

#include <type_traits>

#include "../base/structs_common.hpp"

namespace noarr {

template<class T>
struct to_struct;

template<IsStruct T>
struct to_struct<T> {
	using type = std::remove_cvref_t<T>;
	static constexpr type convert(T t) noexcept { return t; }
};

template<class T>
constexpr auto convert_to_struct(T &&t) noexcept {
	return to_struct<std::remove_cvref_t<T>>::convert(std::forward<T>(t));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TO_STRUCT_HPP
