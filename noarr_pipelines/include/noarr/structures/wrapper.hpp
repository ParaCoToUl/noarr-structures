#ifndef NOARR_WRAPPER_HPP
#define NOARR_WRAPPER_HPP

namespace noarr {

template<typename Structure>
struct wrapper : private Structure {
    constexpr wrapper() = default;
    explicit constexpr wrapper(Structure s) : Structure(s) {}
};

template<typename Structure> 
inline constexpr auto wrap(Structure s) {
    return wrapper<Structure>(s);
}

}

#endif // NOARR_WRAPPER_HPP