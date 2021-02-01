#ifndef NOARR_FUNCS_HPP
#define NOARR_FUNCS_HPP

#include "noarr_structs.hpp"

namespace noarr {

template<char DIM>
struct resize {
    const std::size_t length;
    explicit constexpr resize(std::size_t length) : length{length} {}
    template<typename T>
    auto operator()(vector<DIM, T> v) const {
        return sized_vector<DIM, T>{std::get<0>(v.sub_structures), length};
    }
    template<typename T>
    auto operator()(sized_vector<DIM, T> v) const {
        return sized_vector<DIM, T>{std::get<0>(v.sub_structures), length};
    }
};

template<char DIM, std::size_t L>
struct cresize {
    constexpr cresize() {}
    template<typename T>
    auto operator()(vector<DIM, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
    template<typename T>
    auto operator()(sized_vector<DIM, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
    template<typename T>
    constexpr auto operator()(array<DIM, L, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
};

}

#endif // NOARR_FUNCS_HPP