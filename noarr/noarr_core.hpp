#ifndef NOARR_CORE_HPP
#define NOARR_CORE_HPP

#include "noarr_std_ext.hpp"
#include "noarr_struct_decls.hpp"
#include "noarr_is_struct.hpp"

// TODO?: rework fmapper and getter to check not only if the function is applicable but also whether it returns some bad type (define it as void or bad_value_t or something...)
// TODO: add dim checking (+ for consume_dims(s))
// TODO: add loading and storing to files (binary, json, xml, ...)
// TODO: add struct checkers
// TODO: the piping mechanism should understand dimensions so we don't abuse template sfinae so much
// TODO: use std::integer_sequence and std::index_sequence wherever applicable

namespace noarr {

template<typename S, typename F, typename = void>
struct fmapper;

template<typename S, typename F, typename = void>
struct _fmapper_cond_helper {
    static constexpr bool value = false;
};
template<typename S, typename F, std::size_t Max = std::tuple_size<typename sub_structures<S>::value_type>::value, std::size_t I = Max>
struct construct_builder;

template<typename S, typename F, std::size_t Max, std::size_t I>
struct construct_builder {
    template<std::size_t... IS>
    static constexpr decltype(auto) construct_build(S s, F f) {
        return construct_builder<S, F, Max, I - 1>::template construct_build<I - 1, IS...>(s, f);
    }
};

template<typename S, typename F, std::size_t Max>
struct construct_builder<S, F, Max, 0> {
    template<std::size_t... IS>
    static constexpr decltype(auto) construct_build(S s, F f) {
        return construct_build_last<IS...>(s, f);
    }
    template<std::size_t... IS>
    static constexpr decltype(auto) construct_build_last(S s, F f) {
        return s.construct((std::get<IS>(s.sub_structures()) | f)...);
    }
};

// this explicit instance is here because the more general one makes warnings on structures with zero substructures
template<typename S, typename F>
struct construct_builder<S, F, 0, 0> {
    template<std::size_t... IS>
    static constexpr decltype(auto) construct_build(S s, F f) {
        return construct_build_last<IS...>(s, f);
    }
    template<std::size_t... IS>
    static constexpr decltype(auto) construct_build_last(S s, F) {
        return s.construct();
    }
};

template<typename S, typename F>
struct _fmapper_cond_helper<S, F, void_t<decltype(construct_builder<S, F>::construct_build(std::declval<S>(), std::declval<F>()))>> {
    static constexpr bool value = true;
};

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<_fmapper_cond_helper<std::enable_if_t<!can_apply<F, S>::value, S>, F>::value>>  {
    static constexpr decltype(auto) fmap(S s, F f) {
        return construct_builder<S, F>::construct_build(s, f);
    }
};

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<can_apply<F, S>::value>> {
    static constexpr decltype(auto) fmap(S s, F f) {
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
    static constexpr decltype(auto) get(S s, F f) {
        return std::tuple_cat(
            getter_impl<S, F, std::integral_constant<std::size_t, J + 1>>::get(s, f),
            getter_impl<std::tuple_element_t<J, typename sub_structures<S>::value_type>, F>::get(std::get<J>(sub_structures<S>(s).value), f));
        }
    static constexpr std::size_t count = getter_impl<S, F, std::integral_constant<std::size_t, J + 1>>::count + getter_impl<std::tuple_element_t<J, typename sub_structures<S>::value_type>,F>::count;
};

template<typename S, typename F>
struct getter_impl<S, F, std::integral_constant<std::size_t, 0>, std::enable_if_t<can_apply<F, S>::value>> {
    static constexpr decltype(auto) get(S s, F f) { return std::make_tuple(f(s)); }
    static constexpr std::size_t count = 1;
};

template<typename S, typename F>
struct getter_impl<S, F, std::integral_constant<std::size_t, std::tuple_size<typename sub_structures<S>::value_type>::value>, std::enable_if_t<!can_apply<F, S>::value>> {
    static constexpr decltype(auto) get(S, F) { return std::tuple<>(); }
    static constexpr std::size_t count = 0;
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<can_apply<F, S>::value>> {
    static constexpr decltype(auto) get(S s, F f) { return f(s); }
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<!can_apply<F, S>::value && (getter_impl<S, F>::count == 1)>> {
    static constexpr decltype(auto) get(S s, F f) { return std::get<0>(getter_impl<S, F>::get(s, f)); }
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<!can_apply<F, S>::value && (getter_impl<S, F>::count != 1)>> {
    static_assert(getter_impl<S, F>::count != 0, "getter has to be applicable");
    static_assert(!(getter_impl<S, F>::count > 1), "getter cannot be ambiguous");
    static constexpr void get(S, F) {}
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
    static constexpr decltype(auto) operate(S s, F f) { return fmapper<S, F>::fmap(s, f);  }
};

template<typename F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, get_tag>::value>> {
    template<typename S>
    static constexpr decltype(auto) operate(S s, F f) { return getter<S, F>::get(s, f);  }
};

template<typename F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, top_tag>::value>> {
    template<typename S>
    static constexpr decltype(auto) operate(S s, F f) { return f(s);  }
};

template<typename S, typename F>
inline constexpr auto operator|(S s, F f) -> std::enable_if_t<is_structoid<S>::value, decltype(pipe_decider<F>::template operate<S>(std::declval<S>(), std::declval<F>()))> {
    return pipe_decider<F>::template operate<S>(s, f);
}

template<typename S, typename... FS>
inline constexpr decltype(auto) pipe(S s, FS... funcs);

template<typename S, typename... FS>
struct piper;

template<typename S, typename F, typename... FS>
struct piper<S, F, FS...> {
    static constexpr decltype(auto) pipe(S s, F func, FS... funcs) {
        auto s1 = s | func;
        return piper<remove_cvref<decltype(s1)>, FS...>::pipe(s1, funcs...);
    }
};

template<typename S, typename F>
struct piper<S, F> {
    static constexpr decltype(auto) pipe(S s, F func) {
        return s | func;
    }
};

template<typename S, typename... FS>
inline constexpr decltype(auto) pipe(S s, FS... funcs) {
    return piper<S, FS...>::pipe(s, funcs...);
}

}

#endif // NOARR_CORE_HPP
