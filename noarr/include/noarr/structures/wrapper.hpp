#ifndef NOARR_STRUCTURES_WRAPPER_HPP
#define NOARR_STRUCTURES_WRAPPER_HPP

#include "funcs.hpp"

namespace noarr {

template<typename Structure>
struct wrapper;

template<typename T>
struct _is_cube<wrapper<T>> {
    using type = is_cube<T>;
};

template<typename Structure>
struct wrapper : private Structure {
    constexpr wrapper() = default;
    explicit constexpr wrapper(Structure s) : Structure(s) {}

    template<char Dim>
    constexpr auto set_length(std::size_t length) const {
        return wrap(static_cast<const Structure &>(*this) | noarr::set_length<Dim>(length));
    }

    template<char... Dims, typename... Ts>
    constexpr auto fix(Ts... ts) const {
        return wrap(static_cast<const Structure &>(*this) | noarr::fix<Dims...>(ts...));
    }

    template<char... Dims, typename... Ts>
    constexpr auto offset(Ts... ts) const {
        return static_cast<const Structure &>(*this) | noarr::offset<Dims...>(ts...);
    }

    template<char Dim>
    constexpr auto get_length() const {
        return static_cast<const Structure &>(*this) | noarr::get_length<Dim>();
    }

    constexpr auto get_size() const {
        return static_cast<const Structure &>(*this) | noarr::get_size();
    }

    template<char... Dims, typename V, typename... Ts>
    constexpr auto get_at(V *ptr, Ts... ts) const {
        return static_cast<const Structure &>(*this) | noarr::get_at<Dims...>(ptr, ts...);
    }
};

template<typename Structure> 
inline constexpr auto wrap(Structure s) {
    return wrapper<Structure>(s);
}

}

#endif // NOARR_STRUCTURES_WRAPPER_HPP