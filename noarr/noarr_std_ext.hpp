#ifndef NOARR_STD_EXT_HPP
#define NOARR_STD_EXT_HPP

#include <cstddef>
#include <type_traits>

namespace noarr {

template<class... T>
using void_t = void;

template<class T>
using remove_cvref = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

struct empty_struct_t {
    constexpr empty_struct_t() : _value{} {}
    const char _value[0];
};

template <typename T>
using is_empty = typename std::is_base_of<empty_struct_t, T>;

template<class T>
struct is_array {
    using value_type = bool;
    static constexpr value_type value = false;
};

template<class T, std::size_t N>
struct is_array<T[N]> {
    using value_type = bool;
    static constexpr value_type value = true;
};

} // namespace noarr

#endif // NOARR_STD_EXT_HPP 