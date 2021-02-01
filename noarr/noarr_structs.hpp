#ifndef NOARR_STRUCTS_HPP
#define NOARR_STRUCTS_HPP

#include "noarr_core.hpp"

namespace noarr {

/**
 * @brief The ground structure
 * 
 * @tparam T the stored type
 */
template<typename T>
struct scalar {
    std::tuple<> sub_structures;
    static constexpr dims_impl<> dims = {};
    constexpr scalar() : sub_structures{} {}
    static constexpr auto construct() {
        return scalar<T>{};
    }
    static constexpr std::size_t size() { return sizeof(T); }
};

/**
 * @brief array
 * 
 * @tparam DIM dimmension added by the structure
 * @tparam T substructure type
 */
template<char DIM, std::size_t L, typename T>
struct array {
    const std::tuple<T> sub_structures;
    static constexpr std::size_t length = L;
    static constexpr dims_impl<DIM> dims = {};
    constexpr array() : sub_structures{} {}
    explicit constexpr array(T sub_structure) : sub_structures{std::make_tuple(sub_structure)} {}
    template<typename T2>
    static constexpr auto construct(T2 sub_structure) {
        return array<DIM, L, T2>{sub_structure};
    }

    constexpr std::size_t size() const { return std::get<0>(sub_structures).size() * L; }
    constexpr std::size_t offset(std::size_t i) const { return std::get<0>(sub_structures).size() * i; }
};

/**
 * @brief unsized vector ready to be resized to the desired size, this vector doesn't have size yet
 * 
 * @tparam DIM dimmension added by the structure
 * @tparam T substructure type
 */
template<char DIM, typename T>
struct vector {
    const std::tuple<T> sub_structures;
    static constexpr dims_impl<DIM> dims = {};
    constexpr vector() : sub_structures{} {}
    explicit constexpr vector(T sub_structure) : sub_structures{std::make_tuple(sub_structure)} {}
    template<typename T2>
    static constexpr auto construct(T2 sub_structure) {
        return vector<DIM, T2>{sub_structure};
    }
};

/**
 * @brief sized vector (size reassignable by the resize function)
 * 
 * @tparam DIM dimmension added by the structure
 * @tparam T substructure type
 */
template<char DIM, typename T>
struct sized_vector : vector<DIM, T> {
    const std::size_t length;
    using vector<DIM, T>::dims;
    using vector<DIM, T>::sub_structures;
    constexpr sized_vector(T sub_structure, std::size_t length) : vector<DIM, T>{sub_structure}, length{length} {}
    template<typename T2>
    constexpr auto construct(T2 sub_structure) const {
        return sized_vector<DIM, T2>{sub_structure, length};
    }

    constexpr std::size_t size() const { return std::get<0>(sub_structures).size() * length; }
    constexpr std::size_t offset(std::size_t i) const { return std::get<0>(sub_structures).size() * i; }
};

}

#endif // NOARR_STRUCTS_HPP