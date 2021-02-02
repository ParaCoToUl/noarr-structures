#ifndef NOARR_STRUCTS_HPP
#define NOARR_STRUCTS_HPP

#include "noarr_core.hpp"

namespace noarr {

template<typename T, typename... Ks>
struct _scalar_get_t;

template<typename T>
struct _scalar_get_t<T> {
    using type = T;
};

template<typename T>
struct _scalar_get_t<T, void> {
    using type = T;
};

/**
 * @brief The ground structure
 * 
 * @tparam T the stored type
 */
template<typename T>
struct scalar {
    std::tuple<> sub_structures;
    static constexpr dims_impl<> dims = {};

    template<typename... Ks>
    using get_t = typename _scalar_get_t<T, Ks...>::type;

    constexpr scalar() : sub_structures{} {}
    static constexpr auto construct() {
        return scalar<T>{};
    }
    static constexpr std::size_t size() { return sizeof(T); }
    constexpr std::size_t offset() const { return 0; }
};

template<typename T, typename... Ks>
struct _array_get_t;

template<typename T>
struct _array_get_t<T> {
    using type = T;
};

template<typename T>
struct _array_get_t<T, void> {
    using type = T;
};

template<typename T, std::size_t K>
struct _array_get_t<T, std::integral_constant<std::size_t, K>> {
    using type = T;
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

    template<typename... Ks>
    using get_t = typename _array_get_t<T, Ks...>::type;

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

template<typename T, typename... Ks>
struct _sized_vector_get_t;

template<typename T>
struct _sized_vector_get_t<T> {
    using type = T;
};

template<typename T>
struct _sized_vector_get_t<T, void> {
    using type = T;
};

template<typename T, std::size_t K>
struct _sized_vector_get_t<T, std::integral_constant<std::size_t, K>> {
    using type = T;
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

    template<typename... Ks>
    using get_t = typename _sized_vector_get_t<T, Ks...>::type;

    constexpr sized_vector(T sub_structure, std::size_t length) : vector<DIM, T>{sub_structure}, length{length} {}
    template<typename T2>
    constexpr auto construct(T2 sub_structure) const {
        return sized_vector<DIM, T2>{sub_structure, length};
    }

    constexpr std::size_t size() const { return std::get<0>(sub_structures).size() * length; }
    constexpr std::size_t offset(std::size_t i) const { return std::get<0>(sub_structures).size() * i; }
};

template<typename T, typename... Ks>
struct _fixed_dim_get_t;

template<typename T>
struct _fixed_dim_get_t<T> {
    using type = T;
};

template<typename T>
struct _fixed_dim_get_t<T, void> {
    using type = T;
};

/**
 * @brief fixed dimension, carries a single sub_structure offset by a certain value
 * 
 * @tparam T substructure type
 */
template<typename T>
struct fixed_dim {
    const std::tuple<T> sub_structures;
    const std::size_t offset_;
    static constexpr dims_impl<> dims = {};

    template<typename... Ks>
    using get_t = typename _fixed_dim_get_t<T, Ks...>::type;

    constexpr fixed_dim(T sub_structure, std::size_t offset) : sub_structures{sub_structure}, offset_{offset} {}
    template<typename T2>
    constexpr auto construct(T2 sub_structure) const {
        return fixed_dim<T2>{sub_structure, offset_};
    }

    constexpr std::size_t size() const { return std::get<0>(sub_structures).size() + offset_; }
    constexpr std::size_t offset() const { return offset_; }
};

}

#endif // NOARR_STRUCTS_HPP