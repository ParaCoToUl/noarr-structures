#ifndef NOARR_CORE_HPP
#define NOARR_CORE_HPP

#include <tuple>
#include <cassert>

#include "noarr_std_ext.hpp"

// TODO?: rework fmapper and getter to check not only if the function is applicable but also whether it returns some bad type (define it as void or bad_value_t or something...)
// TODO: add dim checking (+ for consume_dims(s))
// TODO: add loading and storing to files (binary, json, xml, ...) maybe mangling? (but a special kind of mangling)
// TODO: add struct checkers
// TODO: add maximum index to each struct somehow

namespace noarr {

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
    static constexpr std::tuple<> value = std::tuple<>{};
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
    explicit constexpr sub_structures(T t) : value{t.sub_structures()} {}
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

template<typename S, typename F>
inline constexpr auto operator%(S s, F f);

template<typename S, typename F, typename = void>
struct fmapper;

template<typename S, typename F, typename = void>
struct can_apply;

template<typename F, typename S, typename>
struct can_apply { static constexpr bool value = false; };

template<typename F, typename S>
struct can_apply<F, S, void_t<decltype(std::declval<F>()(std::declval<S>()))>> { static constexpr bool value = true; };

template<typename S, typename F, typename = void>
struct _fmapper_cond_helper {
    static constexpr bool value = false;
};
template<typename S, typename F, std::size_t Max = std::tuple_size<typename sub_structures<S>::value_type>::value, std::size_t I = Max>
struct construct_builder;

template<typename S, typename F, std::size_t Max, std::size_t I>
struct construct_builder {
    template<std::size_t... IS>
    static constexpr auto construct_build(S s, F f) {
        return construct_builder<S, F, Max, I - 1>::template construct_build<I - 1, IS...>(s, f);
    }
};

template<typename S, typename F, std::size_t Max>
struct construct_builder<S, F, Max, 0> {
    template<std::size_t... IS>
    static constexpr auto construct_build(S s, F f) {
        return construct_build_last<IS...>(s, f);
    }
    template<std::size_t... IS>
    static constexpr auto construct_build_last(S s, F f) {
        return s.construct((std::get<IS>(s.sub_structures()) % f)...);
    }
};

// this explicit instance is here because the more general one makes warnings on structures with zero substructures
template<typename S, typename F>
struct construct_builder<S, F, 0, 0> {
    template<std::size_t... IS>
    static constexpr auto construct_build(S s, F f) {
        return construct_build_last<IS...>(s, f);
    }
    template<std::size_t... IS>
    static constexpr auto construct_build_last(S s, F) {
        return s.construct();
    }
};

template<typename S, typename F>
struct _fmapper_cond_helper<S, F, void_t<decltype(construct_builder<S, F>::construct_build(std::declval<S>(), std::declval<F>()))>> {
    static constexpr bool value = true;
};

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<_fmapper_cond_helper<std::enable_if_t<!can_apply<F, S>::value, S>, F>::value>>  {
    static constexpr auto fmap(S s, F f) {
        return construct_builder<S, F>::construct_build(s, f);
    }
};

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<can_apply<F, S>::value>> {
    static constexpr auto fmap(S s, F f) {
        return f(s);
    }
};

// TODO: add context
template<typename S, typename F, typename = void>
struct getter;

template<typename S, typename F, typename J = std::integral_constant<std::size_t, 0>, typename = void>
struct getter_impl;

template<typename S, typename F, std::size_t J>
struct getter_impl<S, F, std::integral_constant<std::size_t, J>, std::enable_if_t<!can_apply<F, S>::value>> {
    static constexpr auto get(S s, F f) {
        return std::tuple_cat(
            getter_impl<S, F, std::integral_constant<std::size_t, J + 1>>::get(s, f),
            getter_impl<std::tuple_element_t<J, typename sub_structures<S>::value_type>, F>::get(std::get<J>(sub_structures<S>(s).value), f));
        }
    static constexpr std::size_t count = getter_impl<S, F, std::integral_constant<std::size_t, J + 1>>::count + getter_impl<std::tuple_element_t<J, typename sub_structures<S>::value_type>,F>::count;
};

template<typename S, typename F>
struct getter_impl<S, F, std::integral_constant<std::size_t, 0>, std::enable_if_t<can_apply<F, S>::value>> {
    static constexpr auto get(S s, F f) { return std::make_tuple(f(s)); }
    static constexpr std::size_t count = 1;
};

template<typename S, typename F>
struct getter_impl<S, F, std::integral_constant<std::size_t, std::tuple_size<typename sub_structures<S>::value_type>::value>, std::enable_if_t<!can_apply<F, S>::value>> {
    static constexpr auto get(S, F) { return std::tuple<>{}; }
    static constexpr std::size_t count = 0;
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<can_apply<F, S>::value>> {
    static constexpr auto get(S s, F f) { return f(s); }
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<!can_apply<F, S>::value && (getter_impl<S, F>::count == 1)>> {
    static constexpr auto get(S s, F f) { return std::get<0>(getter_impl<S, F>::get(s, f)); }
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<!can_apply<F, S>::value && (getter_impl<S, F>::count != 1)>> {
    static_assert(getter_impl<S, F>::count != 0, "getter has to be applicable");
    static_assert(!(getter_impl<S, F>::count > 1), "getter cannot be ambiguous");
    static constexpr void get(S, F) {};
};


/* func families */
struct transform_tag;
struct get_tag;
struct top_tag;

using default_trait = transform_tag;

template<typename F, typename = void>
struct func_trait {
    using type = default_trait;
};

template<typename F>
struct func_trait<F, std::enable_if_t<std::is_same<typename F::func_family, transform_tag>::value>> {
    using type = transform_tag;
};

template<typename F>
struct func_trait<F, std::enable_if_t<std::is_same<typename F::func_family, get_tag>::value>> {
    using type = get_tag;
};

template<typename F>
struct func_trait<F, std::enable_if_t<std::is_same<typename F::func_family, top_tag>::value>> {
    using type = top_tag;
};

template<typename F>
using func_trait_t = typename func_trait<F>::type;

template<typename F, typename = void>
struct pipe_decider;

template<typename F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, transform_tag>::value>> {
    template<typename S>
    static constexpr auto operate(S s, F f) { return fmapper<S, F>::fmap(s, f);  }
};

template<typename F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, get_tag>::value>> {
    template<typename S>
    static constexpr auto operate(S s, F f) { return getter<S, F>::get(s, f);  }
};

template<typename F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, top_tag>::value>> {
    template<typename S>
    static constexpr auto operate(S s, F f) { return f(s);  }
};

template<typename S, typename F>
inline constexpr auto operator%(S s, F f) {
    return pipe_decider<F>::template operate<S>(s, f);
}

template<typename S, typename... FS>
inline constexpr auto pipe(S s, FS... funcs);

template<typename S, typename... FS>
struct piper;

template<typename S, typename F, typename... FS>
struct piper<S, F, FS...> {
    static constexpr auto pipe(S s, F func, FS... funcs) {
        auto s1 = s % func;
        return piper<remove_cvref<decltype(s1)>, FS...>::pipe(s1, funcs...);
    }
};

template<typename S, typename F>
struct piper<S, F> {
    static constexpr auto pipe(S s, F func) {
        return s % func;
    }
};

template<typename S, typename... FS>
inline constexpr auto pipe(S s, FS... funcs) {
    return piper<S, FS...>::pipe(s, funcs...);
}

}

#endif // NOARR_CORE_HPP
