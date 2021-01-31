#ifndef NOARR_TYPES_HPP
#define NOARR_TYPES_HPP

#include "noarr_core.hpp"

namespace noarr {

template<typename T>
struct scalar {
    std::tuple<> sub_structures;
    static constexpr dims_impl<> dims = {};
    constexpr scalar() : sub_structures{} {}
    static constexpr scalar<T> construct() {
        return {};
    }
    static constexpr std::size_t size() { return sizeof(T); }
};

template<char DIM, typename T>
struct vector {
    std::tuple<T> sub_structures;
    static constexpr dims_impl<DIM> dims = {};
    constexpr vector() : sub_structures{{}} {}
    constexpr vector(T sub_structure) : sub_structures{std::make_tuple(sub_structure)} {}
    template<typename T2>
    static constexpr vector<DIM, T2> construct(T2 sub_structure) {
        return {sub_structure};
    }
};

template<char DIM, std::size_t L, typename T>
struct array {
    std::tuple<T> sub_structures;
    static constexpr std::size_t length = L;
    static constexpr dims_impl<DIM> dims = {};
    constexpr array() : sub_structures{{}} {}
    constexpr array(T sub_structure) : sub_structures{std::make_tuple(sub_structure)} {}
    template<typename T2>
    static constexpr array<DIM, L, T2> construct(T2 sub_structure) {
        return {sub_structure};
    }

    constexpr std::size_t size() { return std::get<0>(sub_structures).size() * L; }
    constexpr std::size_t offset(std::size_t i) { return std::get<0>(sub_structures).size() * i; }
};

template<char DIM, typename T>
struct sized_vector : vector<DIM, T> {
    std::size_t length;
    using get_t = T;
    using vector<DIM, T>::dims;
    using vector<DIM, T>::sub_structures;
    constexpr sized_vector(T sub_structure, std::size_t length) : vector<DIM, T>{sub_structure}, length{length} {}
    template<typename T2>
    constexpr sized_vector<DIM, T2> construct(T2 sub_structure) const {
        return {sub_structure, length};
    }

    constexpr std::size_t size() { return std::get<0>(sub_structures).size() * length; }
    constexpr std::size_t offset(std::size_t i) { return std::get<0>(sub_structures).size() * i; }
};

template<char DIM>
struct resize {
    std::size_t length;
    constexpr resize(std::size_t length) : length{length} {}
    template<typename T>
    constexpr sized_vector<DIM, T> operator()(vector<DIM, T> v) const {
        return {std::get<0>(v.sub_structures), length};
    }
    template<typename T>
    constexpr sized_vector<DIM, T> operator()(sized_vector<DIM, T> v) const {
        return {std::get<0>(v.sub_structures), length};
    }
};

}

#endif // NOARR_TYPES_HPP