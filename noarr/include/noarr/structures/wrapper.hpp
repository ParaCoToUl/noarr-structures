#ifndef NOARR_STRUCTURES_WRAPPER_HPP
#define NOARR_STRUCTURES_WRAPPER_HPP

#include "funcs.hpp"

namespace noarr {

/**
 * @brief wraps 
 * 
 * @tparam Structure 
 */
template<typename Structure>
class wrapper;

template<typename T>
struct _is_cube<wrapper<T>> {
    using type = is_cube<T>;
};

template<typename Structure>
class wrapper : private contain<Structure> {
    using base = contain<Structure>;

public:
    constexpr wrapper() = default;
    explicit constexpr wrapper(Structure s) : base(s) {}

    template<char Dim>
    constexpr auto set_length(std::size_t length) const {
        return wrap(base::template get<0>() | noarr::set_length<Dim>(length));
    }

    template<char... Dims, typename... Ts>
    constexpr auto fix(Ts... ts) const {
        return wrap(base::template get<0>() | noarr::fix<Dims...>(ts...));
    }

    template<char... Dims, typename... Ts>
    constexpr auto offset(Ts... ts) const {
        return base::template get<0>() | noarr::offset<Dims...>(ts...);
    }

    template<char Dim>
    constexpr auto get_length() const {
        return base::template get<0>() | noarr::get_length<Dim>();
    }

    constexpr auto get_size() const {
        return base::template get<0>() | noarr::get_size();
    }

    template<char... Dims, typename V, typename... Ts>
    constexpr decltype(auto) get_at(V *ptr, Ts... ts) const {
        return base::template get<0>() | noarr::get_at<Dims...>(ptr, ts...);
    }

    constexpr auto unwrap() const {
        return base::template get<0>();
    }
};

template<typename Structure> 
inline constexpr wrapper<Structure> wrap(Structure s) {
    return wrapper<Structure>(s);
}

}

#endif // NOARR_STRUCTURES_WRAPPER_HPP
