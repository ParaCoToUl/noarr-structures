#ifndef NOARR_STRUCTURES_PIPES_HPP
#define NOARR_STRUCTURES_PIPES_HPP

#include "std_ext.hpp"
#include "struct_decls.hpp"
#include "is_struct.hpp"

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
 * @tparam class: placeholder type
 */
template<class F, class = void>
struct func_trait {
	using type = default_trait;
};

template<class F>
struct func_trait<F, std::enable_if_t<std::is_same<typename F::func_family, transform_tag>::value>> {
	using type = transform_tag;
};

template<class F>
struct func_trait<F, std::enable_if_t<std::is_same<typename F::func_family, get_tag>::value>> {
	using type = get_tag;
};

template<class F>
struct func_trait<F, std::enable_if_t<std::is_same<typename F::func_family, top_tag>::value>> {
	using type = top_tag;
};

template<class F>
using func_trait_t = typename func_trait<F>::type;

/**
 * @brief returns whether the function `F` is applicable to the structure `S` according to its `::can_apply<S>::value` (defaults to true if not present)
 * 
 * @tparam F: the tested function
 * @tparam S: the structure
 * @tparam class: placeholder type
 */
template<class F, class S, class = void>
struct get_applicability : std::true_type {};

template<class F, class S>
struct get_applicability<F, S, void_t<typename F::template can_apply<S>>> : F::template can_apply<S> {};

/**
 * @brief returns whether the function `F` is applicable to the structure `S`, also honoring `get_applicability`
 * 
 * @tparam S: the structure
 * @tparam F: the tested function
 * @tparam class: placeholder type
 */
template<class S, class F, class = void>
struct can_apply : std::false_type {};

template<class F, class S>
struct can_apply<F, S, void_t<decltype(std::declval<F>()(std::declval<S>()))>> : get_applicability<F, S> {};

namespace helpers {

/**
 * @brief applies the function `F` to the structure `S` or, if not possible, attempts to do so on its substructures recursively
 * 
 * @tparam S: the structure to be mapped
 * @tparam F: the mapping function
 * @tparam class: placeholder type
 */
template<class S, class F, class = void>
struct fmapper;

/**
 * @brief A function used in `fmapper` which reconstructs the *unmapped* structure `S` (with mapped substructures) according to the mapping described by the function `F`
 * 
 * @tparam S: the structure to be reconstructed after the substructures are mapped
 * @tparam F: the mapping function
 * @tparam Max: the ceiling of iteration throught substructures in the `construct` function
 * @tparam I: the iterating variable which iterates throught substructures
 */
template<class S, class F, std::size_t Max = std::tuple_size<typename sub_structures<S>::value_type>::value, std::size_t I = Max>
struct construct_builder;

template<class S, class F>
struct fmapper<S, F, std::enable_if_t<!can_apply<F, S>::value>> {
	static constexpr auto fmap(S s, F f) noexcept {
		return construct_builder<S, F>::construct_build(s, f);
	}
};

template<class S, class F>
struct fmapper<S, F, std::enable_if_t<can_apply<F, S>::value>> {
	static constexpr auto fmap(S s, F f) noexcept {
		return f(s);
	}
};

/**
 * @brief gets a value from the structure `S` using a function `F`. If `F` cannot be applied to `S`, application to substructures is attempted (stopping at the first structure `F` is applicable to). If there are multiple branches throught the substructure tree ending in application the getter fails
 * 
 * @tparam S: the structure
 * @tparam F: the function which retrieves a value from the structure
 * @tparam class: a placeholder type
 */
template<class S, class F, class = void>
struct getter;

template<class S, class F, class J = std::integral_constant<std::size_t, 0>, class = void>
struct getter_impl;

template<class S, class F, std::size_t J>
struct getter_impl<S, F, std::integral_constant<std::size_t, J>, std::enable_if_t<!can_apply<F, S>::value>> {
	static constexpr auto get(S s, F f) noexcept {
		return std::tuple_cat(
			getter_impl<S, F, std::integral_constant<std::size_t, J + 1>>::get(s, f),
			getter_impl<std::tuple_element_t<J, typename sub_structures<S>::value_type>, F>::get(std::get<J>(sub_structures<S>(s).value), f));
		}
	static constexpr std::size_t count = getter_impl<S, F, std::integral_constant<std::size_t, J + 1>>::count + getter_impl<std::tuple_element_t<J, typename sub_structures<S>::value_type>,F>::count;
};

template<class S, class F>
struct getter_impl<S, F, std::integral_constant<std::size_t, 0>, std::enable_if_t<can_apply<F, S>::value>> {
	static constexpr auto get(S s, F f) noexcept { return std::make_tuple(f(s)); }
	static constexpr std::size_t count = 1;
};

template<class S, class F>
struct getter_impl<S, F, std::integral_constant<std::size_t, std::tuple_size<typename sub_structures<S>::value_type>::value>, std::enable_if_t<!can_apply<F, S>::value>> {
	static constexpr auto get(S, F) noexcept { return std::tuple<>(); }
	static constexpr std::size_t count = 0;
};

template<class S, class F>
struct getter<S, F, std::enable_if_t<can_apply<F, S>::value>> {
	static constexpr auto get(S s, F f) noexcept { return f(s); }
};

template<class S, class F>
struct getter<S, F, std::enable_if_t<!can_apply<F, S>::value && (getter_impl<S, F>::count == 1)>> {
	static constexpr auto get(S s, F f) noexcept { return std::get<0>(getter_impl<S, F>::get(s, f)); }
};

template<class S, class F>
struct getter<S, F, std::enable_if_t<!can_apply<F, S>::value && (getter_impl<S, F>::count != 1)>> {
	static_assert(getter_impl<S, F>::count != 0, "getter has to be applicable");
	static_assert(!(getter_impl<S, F>::count > 1), "getter cannot be ambiguous");
	static constexpr void get(S, F) noexcept {}
};

template<class F, class = void>
struct pipe_decider;

/**
 * @brief decided whether perform `fmapper`, `getter` or simple application according to `func_trait`
 * 
 */
template<class F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, transform_tag>::value>> {
	template<class S>
	static constexpr auto operate(S s, F f) noexcept { return fmapper<S, F>::fmap(s, f); }
};

template<class F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, get_tag>::value>> {
	template<class S>
	static constexpr decltype(auto) operate(S s, F f) noexcept { return getter<S, F>::get(s, f); }
};

template<class F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, top_tag>::value>> {
	template<class S>
	static constexpr decltype(auto) operate(S s, F f) noexcept { return f(s); }
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
template<class S, class F>
constexpr auto operator|(S s, F f) noexcept ->
std::enable_if_t<is_structoid<std::enable_if_t<std::is_class<S>::value, S>>::value, decltype(helpers::pipe_decider<F>::template operate<S>(std::declval<S>(), std::declval<F>()))> {
	return helpers::pipe_decider<F>::template operate<S>(s, f);
}

namespace helpers {

template<class S, class F, std::size_t Max, std::size_t I>
struct construct_builder {
	template<std::size_t... IS>
	static constexpr auto construct_build(S s, F f) noexcept {
		return construct_builder<S, F, Max, I - 1>::template construct_build<I - 1, IS...>(s, f);
	}
};

template<class S, class F, std::size_t Max>
struct construct_builder<S, F, Max, 0> {
	template<std::size_t... IS>
	static constexpr auto construct_build(S s, F f) noexcept {
		return construct_build_last<IS...>(s, f);
	}
	template<std::size_t... IS>
	static constexpr auto construct_build_last(S s, F f) noexcept {
		return s.construct((std::get<IS>(s.sub_structures()) | f)...);
	}
};

// this explicit instance is here because the more general one makes warnings on structures with zero substructures
template<class S, class F>
struct construct_builder<S, F, 0, 0> {
	template<std::size_t... IS>
	static constexpr auto construct_build(S s, F f) noexcept {
		return construct_build_last<IS...>(s, f);
	}
	template<std::size_t... IS>
	static constexpr auto construct_build_last(S s, F) noexcept {
		return s.construct();
	}
};

template<class S, class... FS>
struct piper;

template<class S, class F, class... FS>
struct piper<S, F, FS...> {
	static constexpr decltype(auto) pipe(S s, F func, FS... funcs) noexcept {
		return piper<remove_cvref<decltype(s | func)>, FS...>::pipe(s | func, funcs...);
	}
};

template<class S, class F>
struct piper<S, F> {
	static constexpr decltype(auto) pipe(S s, F func) noexcept {
		return s | func;
	}
};

} // namespace helpers

/**
 * @brief performs a `fmapper::fmap`, `getter::get` or a simple application of `F` and subsequent `FS` to `S` according to `pipe_decider`
 * 
 * @param s: the structure
 * @param funcs: applied functions (consecutively)
 */
template<class S, class... FS>
constexpr decltype(auto) pipe(S s, FS... funcs) noexcept {
	return helpers::piper<S, FS...>::pipe(s, funcs...);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_PIPES_HPP
