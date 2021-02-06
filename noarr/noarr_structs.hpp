#ifndef NOARR_STRUCTS_HPP
#define NOARR_STRUCTS_HPP

#include "noarr_core.hpp"
#include "noarr_struct_desc.hpp"

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
    std::tuple<> sub_structures() const { return {}; }
    static constexpr dims_impl<> dims = {};
    using desc = struct_desc<
        integral_pack<char, 's', 'c', 'a', 'l', 'a', 'r'>,
        dims_impl<>,
        dims_impl<>,
        type_param<T>>;

    template<typename... Ks>
    using get_t = typename _scalar_get_t<T, Ks...>::type;

    constexpr scalar() {}
    static constexpr auto construct() {
        return scalar<T>{};
    }
    static constexpr std::size_t size() { return sizeof(T); }
    constexpr std::size_t offset() const { return 0; }
};

// TODO: finish tuple description
/**
 * @brief tuple
 * 
 * @tparam DIM dimmension added by the structure
 * @tparam T,Ts... substructure types
 */
template<char DIM, typename... Ts>
struct tuple;

template<typename TUPLE, std::size_t I>
struct tuple_part;

template<typename T, typename... Ks>
struct _tuple_get_t;

template<char DIM, typename T, typename... Ts, std::size_t I, std::size_t K>
struct _tuple_get_t<tuple_part<tuple<DIM, T, Ts...>, I>, std::integral_constant<std::size_t, K>> {
    using type = typename _tuple_get_t<tuple_part<tuple<DIM, Ts...>, I + 1>, std::integral_constant<std::size_t, K - 1>>::type;
};

template<char DIM, typename T, typename... Ts, std::size_t I>
struct _tuple_get_t<tuple_part<tuple<DIM, T, Ts...>, I>, std::integral_constant<std::size_t, 0>> {
    using type = T;
};

template<char DIM, typename... Ts, typename T, std::size_t I>
struct tuple_part<tuple<DIM, T, Ts...>, I> : private tuple_part<tuple<DIM, Ts...>, I + 1>, private T {
    constexpr tuple_part() : tuple_part<tuple<DIM, Ts...>, I + 1>{}, T{} {}
    constexpr tuple_part(T t, Ts... ts) : tuple_part<tuple<DIM, Ts...>, I + 1>{ts...}, T{t} {}
    constexpr auto sub_structures() const {
        return std::tuple_cat(std::tuple<T>{static_cast<const T&>(*this)}, tuple_part<tuple<DIM, Ts...>, I + 1>::sub_structures());
    }
};

template<char DIM, typename T, std::size_t I>
struct tuple_part<tuple<DIM, T>, I> : private T {
    constexpr tuple_part() : T{} {}
    explicit constexpr tuple_part(T t) : T{t} {}
    constexpr auto sub_structures() const {
        return std::tuple<T>{static_cast<const T&>(*this)};
    }
};

template<typename T, std::size_t I = std::tuple_size<T>::value>
struct _tuple_size_getter;

template<typename T, std::size_t I>
struct _tuple_size_getter {
    template<std::size_t... IDXs>
    static constexpr std::size_t size(T t) {
        return _tuple_size_getter<T, I - 1>::template size<I - 1, IDXs...>(t);
    }
};

template<typename T>
struct _tuple_size_getter<T, 0> {
    template<typename... Args>
    static constexpr std::size_t sum(std::size_t arg, Args... args) {
        return arg + sum(args...);
    }
    static constexpr std::size_t sum(std::size_t arg) {
        return arg;
    }
    template<std::size_t... IDXs>
    static constexpr std::size_t size(T t) {
        return sum(std::get<IDXs>(t).size()...);
    }
};

template<char DIM, typename T, typename... Ts>
struct tuple<DIM, T, Ts...> : private tuple_part<tuple<DIM, T, Ts...>, 0> {
    static constexpr dims_impl<DIM> dims = {};
    constexpr std::tuple<T, Ts...> sub_structures() const { return tuple_part<tuple<DIM, T, Ts...>, 0>::sub_structures(); }
    using desc = struct_desc<
        integral_pack<char, 't', 'u', 'p', 'l', 'e'>,
        dims_impl<DIM>,
        dims_impl<>,
        type_param<T>,
        type_param<Ts>...>;

    template<typename... Ks>
    using get_t = typename _tuple_get_t<tuple_part<tuple<DIM, T, Ts...>, 0>, Ks...>::type;

    constexpr tuple() : tuple_part<tuple<DIM, T, Ts...>, 0>{} {}
    constexpr tuple(T ss, Ts... sss) : tuple_part<tuple<DIM, T, Ts...>, 0>{ss, sss...} {}
    template<typename T2, typename... T2s>
    constexpr auto construct(T2 ss, T2s... sss) const {
        return tuple<DIM, T2, T2s...>{ss, sss...};
    }

    constexpr std::size_t size() const {
        return _tuple_size_getter<remove_cvref<decltype(sub_structures())>>::size(sub_structures());
    }
    template<std::size_t i>
    constexpr std::size_t offset() const {
        return _tuple_size_getter<remove_cvref<decltype(sub_structures())>, i>::size(sub_structures());
    }
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

// TODO: finish array description
/**
 * @brief array
 * 
 * @tparam DIM dimmension added by the structure
 * @tparam T substructure type
 */
template<char DIM, std::size_t L, typename T>
struct array : private T {
    static constexpr std::size_t length = L;
    static constexpr dims_impl<DIM> dims = {};
    constexpr std::tuple<T> sub_structures() const { return {static_cast<const T&>(*this)}; }
    using desc = struct_desc<
        integral_pack<char, 'a', 'r', 'r', 'a', 'y'>,
        dims_impl<DIM>,
        dims_impl<>,
        value_param<std::size_t, L>,
        type_param<T>>;

    template<typename... Ks>
    using get_t = typename _array_get_t<T, Ks...>::type;

    constexpr array() : T{} {}
    explicit constexpr array(T sub_structure) : T{sub_structure} {}
    template<typename T2>
    constexpr auto construct(T2 sub_structure) const {
        return array<DIM, L, T2>{sub_structure};
    }

    constexpr std::size_t size() const { return static_cast<const T&>(*this).size() * L; }
    constexpr std::size_t offset(std::size_t i) const { return static_cast<const T&>(*this).size() * i; }
};

/**
 * @brief unsized vector ready to be resized to the desired size, this vector doesn't have size yet
 * 
 * @tparam DIM dimmension added by the structure
 * @tparam T substructure type
 */
template<char DIM, typename T>
struct vector : private T {
    static constexpr dims_impl<DIM> dims = {};
    constexpr std::tuple<T> sub_structures() const { return {static_cast<const T&>(*this)}; }
    using desc = struct_desc<
        integral_pack<char, 'v', 'e', 'c', 't', 'o', 'r'>,
        dims_impl<DIM>,
        dims_impl<>,
        type_param<T>>;

    constexpr vector() : T{} {}
    explicit constexpr vector(T sub_structure) : T{sub_structure} {}
    template<typename T2>
    constexpr auto construct(T2 sub_structure) const {
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
struct sized_vector : private vector<DIM, T> {
    const std::size_t length;
    using vector<DIM, T>::dims;
    using vector<DIM, T>::sub_structures;
    using desc = struct_desc<
        integral_pack<char, 's', 'i', 'z', 'e', 'd', '_', 'v', 'e', 'c', 't', 'o', 'r'>,
        dims_impl<DIM>,
        dims_impl<>,
        type_param<T>>;

    template<typename... Ks>
    using get_t = typename _sized_vector_get_t<T, Ks...>::type;

    constexpr sized_vector(T sub_structure, std::size_t length) : vector<DIM, T>{sub_structure}, length{length} {}
    template<typename T2>
    constexpr auto construct(T2 sub_structure) const {
        return sized_vector<DIM, T2>{sub_structure, length};
    }

    constexpr std::size_t size() const { return std::get<0>(sub_structures()).size() * length; }
    constexpr std::size_t offset(std::size_t i) const { return std::get<0>(sub_structures()).size() * i; }
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
 * @brief fixed dimension, carries a single sub_structure with a fixed index
 * 
 * @tparam T substructure type
 */
template<char DIM, typename T>
struct fixed_dim : private T {
    const std::size_t idx_;
    static constexpr dims_impl<DIM> dims = {};
    static constexpr dims_impl<DIM> consume_dims = {};
    constexpr auto sub_structures() const { return static_cast<const T&>(*this).sub_structures(); }
    using desc = struct_desc<
        integral_pack<char, 'f', 'i', 'x', 'e', 'd', '_', 'd', 'i', 'm'>,
        dims_impl<DIM>,
        dims_impl<>,
        type_param<T>>;

    template<typename... Ks>
    using get_t = typename _fixed_dim_get_t<T, Ks...>::type;

    constexpr fixed_dim(T sub_structure, std::size_t idx) : T{sub_structure}, idx_{idx} {}
    template<typename T2>
    constexpr auto construct(T2 sub_structure) const {
        return fixed_dim<DIM, decltype(T::construct(sub_structure))>{
            static_cast<const T&>(*this).construct(sub_structure), idx_};
    }

    constexpr std::size_t size() const { return static_cast<const T&>(*this).size(); }
    constexpr std::size_t offset() const { return static_cast<const T&>(*this).offset(idx_); }
};

}

#endif // NOARR_STRUCTS_HPP
