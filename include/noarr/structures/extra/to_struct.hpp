#ifndef NOARR_STRUCTURES_TO_STRUCT_HPP
#define NOARR_STRUCTURES_TO_STRUCT_HPP

#include "../base/structs_common.hpp"

namespace noarr {

template<class T, class = void>
struct to_struct;

template<class T>
struct to_struct<T, std::enable_if_t<is_struct<T>::value>> {
	using type = T;
	static constexpr T convert(T t) noexcept { return t; }
};

} // namespace noarr

#endif // NOARR_STRUCTURES_TO_STRUCT_HPP
