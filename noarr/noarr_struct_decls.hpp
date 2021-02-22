#ifndef NOARR_STRUCT_DECLS_HPP
#define NOARR_STRUCT_DECLS_HPP

#include <tuple>

#include "noarr_std_ext.hpp"

namespace noarr {

// TODO: add a way to get Params
// TODO: split type_param into struct_param and scalar_param

template<typename Name, typename Dims, typename ADims, typename... Params>
struct struct_description {
    using name = Name;
    using dims = Dims;
    using adims = ADims;
    using description = struct_description;
};

template<typename>
struct type_param;

template<typename T, T V>
struct value_param;

template<typename T, typename = void>
struct get_struct_desc;

template<typename T>
using get_struct_desc_t = typename get_struct_desc<T>::type;

// TODO: check if integral_pack
template<typename T>
struct get_struct_desc<T, void_t<typename T::description>> {
    using type = typename T::description;
};

/**
 * @brief retrieves sub_structures from the given structure
 * 
 * @return value of type value_type will hold the list of sub_structures in a tuple
 * @tparam T the type of the given structure
 */
template<typename T, typename = void>
struct sub_structures {
    explicit constexpr sub_structures() = default;
    explicit constexpr sub_structures(T) {}
    using value_type = std::tuple<>;
    static constexpr std::tuple<> value = std::tuple<>();
};

template<typename T, typename = void>
struct _sub_structures_is_static {
    static constexpr bool value = false;
};

template<typename T>
struct _sub_structures_is_static<T, void_t<decltype(T::sub_structures())>> {
    static constexpr bool value = true;
};

// TODO: check if tuple
template<typename T>
struct sub_structures<T, std::enable_if_t<_sub_structures_is_static<T>::value>> {
    explicit constexpr sub_structures() = default;
    explicit constexpr sub_structures(T) {}
    using value_type = remove_cvref<decltype(T::sub_structures())>;
    static constexpr value_type value = T::sub_structures();
};

template<typename T>
struct sub_structures<T, std::enable_if_t<!_sub_structures_is_static<T>::value, void_t<decltype(std::declval<T>().sub_structures())>>> {
    explicit constexpr sub_structures() = delete;
    explicit constexpr sub_structures(T t) : value(t.sub_structures()) {}
    using value_type = remove_cvref<decltype(std::declval<T>().sub_structures())>;
    value_type value;
};

/**
 * @brief The type that holds all the dimensions of a structure
 * 
 * @tparam Dims the dimensions
 */
template<char... Dims>
using dims_impl = char_pack<Dims...>;

template<typename T>
using get_dims = typename T::description::dims;
// TODO: implement the recursive version using sub_structures

template<typename T, std::size_t I = std::tuple_size<typename sub_structures<T>::value_type>::value>
struct _construct;

template<typename T, std::size_t I>
struct _construct {
    template<std::size_t... IS, typename... TS>
    static constexpr auto construct(T t, std::tuple<TS...> sub_structures){
        return _construct<T, I - 1>::template construct<I - 1, IS...>(t, sub_structures);
    }
};

template<typename T>
struct _construct<T, 0> {
    template<std::size_t... IS, typename... TS>
    static constexpr auto construct(T t, std::tuple<TS...> sub_structures) {
        return t.construct(std::get<IS>(sub_structures)...);
    }

    template<std::size_t... IS>
    static constexpr auto construct(T t, std::tuple<>) {
        return t.construct();
    }
};

template<typename T, typename... TS>
inline constexpr auto construct(T t, TS... ts) {
    return t.construct(ts...);
}

template<typename T, typename... TS>
inline constexpr auto construct(T t, std::tuple<TS...> ts) {
    return _construct<T>::construct(t, ts);
}

}

#endif // NOARR_STRUCT_DECLS_HPP