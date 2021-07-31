#ifndef NOARR_STRUCTURES_CORE_HPP
#define NOARR_STRUCTURES_CORE_HPP

#include "std_ext.hpp"
#include "struct_decls.hpp"
#include "is_struct.hpp"

// TODO: add loading and storing to files (binary, json?, xml?, ...)
// TODO: add a method for structures that configures the struct => this will be reflected by 2nd serialization

namespace noarr {

/* func families */

/**
 * @brief this family is for functions that map structures to another structures by performing transformations of certain substructures
 * 
 */
struct transform_tag;

/**
 * @brief this family is for functions that retrieve a value from a structure
 * 
 */
struct get_tag;

/**
 * @brief this family is for functions that map structures to another structures by performing a transformation of the topmost structure (opposing `transform_tag`)
 * 
 */
struct top_tag;

using default_trait = transform_tag;

/**
 * @brief retrieves a family tag from a function `F`
 * 
 * @tparam F: the function 
 * @tparam typename 
 */
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

/**
 * @brief returns whether the function `F` is applicable to the structure `S` according to its `::can_apply<S>::value` (defaults to true if not present)
 * 
 * @tparam F: the tested function
 * @tparam S: the structure
 * @tparam typename: placeholder type
 */
template<typename F, typename S, typename = void>
struct get_applicability {
    static constexpr bool value = true;
};

template<typename F, typename S>
struct get_applicability<F, S, void_t<typename F::template can_apply<S>>> {
    static constexpr bool value = F::template can_apply<S>::value;
};

/**
 * @brief returns whether the function `F` is applicable to the structure `S`, also honoring `get_applicability`
 * 
 * @tparam S: the structure
 * @tparam F: the tested function
 * @tparam typename: placeholder type
 */
template<typename S, typename F, typename = void>
struct can_apply {
    static constexpr bool value = false;
};

template<typename F, typename S>
struct can_apply<F, S, void_t<decltype(std::declval<F>()(std::declval<S>()))>> {
    static constexpr bool value = get_applicability<F, S>::value;
};

namespace helpers {

/**
 * @brief applies the function `F` to the structure `S` or, if not possible, attempts to do so on its substructures recursively
 * 
 * @tparam S: the structure to be mapped
 * @tparam F: the mapping function
 * @tparam typename: placeholder type
 */
template<typename S, typename F, typename = void>
struct fmapper;

/**
 * @brief A function used in `fmapper` which reconstructs the *unmapped* structure `S` (with mapped substructures) according to the mapping described by the function `F`
 * 
 * @tparam S: the structure to be reconstructed after the substructures are mapped
 * @tparam F: the mapping function
 * @tparam Max: the ceiling of iteration throught substructures in the `construct` function
 * @tparam I: the iterating variable which iterates throught substructures 
 */
template<typename S, typename F, std::size_t Max = std::tuple_size<typename sub_structures<S>::value_type>::value, std::size_t I = Max>
struct construct_builder;

template<typename S, typename F, typename = void>
struct fmapper_cond_helper {
    static constexpr bool value = false;
};

template<typename S, typename F>
struct fmapper_cond_helper<S, F, void_t<decltype(construct_builder<S, F>::construct_build(std::declval<S>(), std::declval<F>()))>> {
    static constexpr bool value = true;
};

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
struct fmapper<S, F, std::enable_if_t<fmapper_cond_helper<std::enable_if_t<!can_apply<F, S>::value, S>, F>::value>>  {
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

/**
 * @brief gets a value from the structure `S` using a function `F`. If `F` cannot be applied to `S`, application to substructures is attempted (stopping at the first structure `F` is applicable to). If there are multiple branches throught the substructure tree ending in application the getter fails
 * 
 * @tparam S: the structure
 * @tparam F: the function which retrieves a value from the structure
 * @tparam typename: a placeholder type
 */
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

template<typename F, typename = void>
struct pipe_decider;

/**
 * @brief decided whether perform `fmapper`, `getter` or simple application according to `func_trait`
 * 
 */
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

} // namespace helpers

/**
 * @brief performs a `fmapper::fmap`, `getter::get` or a simple application of `F` to `S` according to `pipe_decider`
 * 
 * @tparam S: the structure type
 * @tparam F: the function type
 * @param s: the structure
 * @param f: the function
 * @return the result of the piping
 */
template<typename S, typename F>
inline constexpr auto operator|(S s, F f) ->
std::enable_if_t<is_structoid<std::enable_if_t<std::is_class<S>::value, S>>::value, decltype(helpers::pipe_decider<F>::template operate<S>(std::declval<S>(), std::declval<F>()))> {
    return helpers::pipe_decider<F>::template operate<S>(s, f);
}

namespace helpers {

template<typename S, typename... FS>
struct piper;

template<typename S, typename F, typename... FS>
struct piper<S, F, FS...> {
    static constexpr decltype(auto) pipe(S s, F func, FS... funcs) {
        return piper<remove_cvref<decltype(s | func)>, FS...>::pipe(s | func, funcs...);
    }
};

template<typename S, typename F>
struct piper<S, F> {
    static constexpr decltype(auto) pipe(S s, F func) {
        return s | func;
    }
};

} // namespace helpers

/**
 * @brief performs a `fmapper::fmap`, `getter::get` or a simple application of `F` and subsequent `FS` to `S` according to `pipe_decider`
 * 
 * @tparam S 
 * @tparam FS 
 * @param s 
 * @param funcs 
 * @return constexpr decltype(auto) 
 */
template<typename S, typename... FS>
inline constexpr decltype(auto) pipe(S s, FS... funcs) {
    return helpers::piper<S, FS...>::pipe(s, funcs...);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_CORE_HPP
