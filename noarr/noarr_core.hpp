#ifndef NOARR_CORE_HPP
#define NOARR_CORE_HPP

#include <tuple>
#include <cassert>

#include "noarr_std_ext.hpp"

// TODO?: rework fmapper and getter to check not only if the function is applicable but also whether it returns some bad type (define it as void or bad_value_t or something...)

namespace noarr {

template<char... DIMs>
struct dims_impl;

template<>
struct dims_impl<> {};

template<char DIM, char... DIMs>
struct dims_impl<DIM, DIMs...> : dims_impl<DIMs...> {
    using value_type = char;
    static constexpr value_type value = DIM;
};

template<class T>
struct dims_length;

template<>
struct dims_length<dims_impl<>> {
    using value_type = std::size_t;
    static constexpr value_type value = 0UL;
};

template<char DIM, char... DIMs>
struct dims_length<dims_impl<DIM, DIMs...>> {
    static constexpr std::size_t value = dims_length<dims_impl<DIMs...>>::value + 1UL;
};

template<typename T, typename = void>
struct sub_structures {
    explicit sub_structures(T) {}
    using value_type = std::tuple<>;
    static constexpr std::tuple<> value = std::tuple<>{};
};

// TODO: check if tuple
template<typename T>
struct sub_structures<T, void_t<decltype(T::sub_structures)>> {
    explicit constexpr sub_structures(T t) : value {t.sub_structures} {}
    using value_type = remove_cvref<decltype(T::sub_structures)>;
    const value_type value;
};

template<typename T, typename = void>
struct dims { 
    static constexpr dims_impl<> value = {};
};

// TODO: check if of type dims_impl
template<typename T>
struct dims<T, void_t<decltype(T::dims)>> {
    static constexpr auto value = T::dims;
};

/**
 * @brief returns (via value) true if DIMS contain DIM, otherwise returns false
 * 
 * @tparam DIMS is expected of type dims_impl, otherwise remove_cvref<decltype(dims<DIMS>::value)> is applied
 * @tparam DIM the needle DIM
 */
template<class DIMS, char DIM, typename = void>
struct dims_have {
    using value_type = bool;
    static constexpr value_type value = dims_have<remove_cvref<decltype(dims<DIMS>::value)>, DIM>::value;
};

template<char NEEDLE> // HAY is empty
struct dims_have<dims_impl<>, NEEDLE> {
    using value_type = bool;
    static constexpr value_type value = false;
};

template<char NEEDLE, char... HAY_TAIL> // NEEDLE == HAY_HEAD
struct dims_have<dims_impl<NEEDLE, HAY_TAIL...>, NEEDLE> {
    using value_type = bool;
    static constexpr value_type value = true;
};

template<char NEEDLE, char HAY_HEAD, char... HAY_TAIL>
struct dims_have<dims_impl<HAY_HEAD, HAY_TAIL...>, NEEDLE, std::enable_if_t<HAY_HEAD != NEEDLE>> {
    using value_type = bool;
    static constexpr value_type value = dims_have<dims_impl<HAY_TAIL...>, NEEDLE>::value;
};

// TODO: implement the recursive version using sub_structures

template<typename S, typename F>
constexpr auto operator%(S s, F f);

template<typename S, typename F, typename = void>
struct fmapper;

template<typename S, typename F, typename = void>
struct can_apply;

struct fmap_traverser;

template<std::size_t I, std::size_t J = 0>
struct fmap_traverser_impl;

template<std::size_t I, std::size_t J>
struct fmap_traverser_impl {
    friend struct fmap_traverser;
    template<std::size_t, std::size_t>
    friend struct fmap_traverser_impl;

private:
    template<typename S, typename F, std::size_t ...Is>
    static constexpr auto sub_fmap_build(S s, F f) {
        return fmap_traverser_impl<I, J + 1>::template sub_fmap_build<S, F, J, Is...>(s, f);
    }
};

template<std::size_t I>
struct fmap_traverser_impl<I, I> {
    friend struct fmap_traverser;
    template<std::size_t, std::size_t>
    friend struct fmap_traverser_impl;

private:
    template<typename S, typename F, std::size_t ...Is>
    static constexpr auto sub_fmap_explicit(S s, F f) {
        return s.construct((std::get<I - 1 - Is>(sub_structures<S>{s}.value) % f) ...);
    }

    template<typename S, typename F, std::size_t ...Is>
    static constexpr auto sub_fmap_build(S s, F f) {
        return sub_fmap_explicit<S, F, Is...>(s, f);
    }
};

template<>
struct fmap_traverser_impl<0, 0> {
private:
    template<typename S, typename F>
    static constexpr auto sub_fmap_explicit(S s, F) {
        return s;
    }
public:
    template<typename S, typename F>
    static constexpr auto sub_fmap_build(S s, F f) {
        return sub_fmap_explicit<S, F>(s, f);
    }
};

struct fmap_traverser {
    template<typename S, typename F>
    static constexpr auto sub_fmap(S s, F f) {
        return fmap_traverser_impl<std::tuple_size<typename sub_structures<S>::value_type>::value>::template sub_fmap_build<S, F>(s, f);
    }
};

template<typename S, typename F, typename>
struct can_apply { static constexpr bool value = false; };

template<typename S, typename F>
struct can_apply<S, F, void_t<decltype(std::declval<F>()(std::declval<S>()))>> { static constexpr bool value = true; };

template<typename S, typename F>
constexpr bool can_apply_v = can_apply<S, F, void>::value;

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<!can_apply_v<S, F>, void_t<decltype(fmap_traverser::sub_fmap(std::declval<S>(), std::declval<F>()))>>>  {
    static constexpr auto fmap(S s, F f) {
        return fmap_traverser::sub_fmap(s, f);
    }
};

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<can_apply_v<S, F>>> {
    static constexpr auto fmap(S s, F f) {
        return f(s);
    }

};

// TODO: add context
template<typename S, typename F, typename = void>
struct getter;

template<typename S, typename F, std::size_t J = 0, typename = void>
struct getter_impl;

template<typename S, typename F, std::size_t J>
struct getter_impl<S, F, J, std::enable_if_t<!can_apply_v<S, F>>> {
    static constexpr auto get(S s, F f) {
        return std::tuple_cat(
            getter_impl<S, F, J + 1>::get(s, f),
            getter_impl<std::tuple_element_t<J, typename sub_structures<S>::value_type>, F>::get(std::get<J>(sub_structures<S>(s).value), f));
        }
    static constexpr std::size_t count = getter_impl<S, F, J + 1>::count + getter_impl<std::tuple_element_t<J, typename sub_structures<S>::value_type>,F>::count;
};

template<typename S, typename F>
struct getter_impl<S, F, 0, std::enable_if_t<can_apply_v<S, F>>> {
    static constexpr auto get(S s, F f) { return std::make_tuple(f(s)); }
    static constexpr std::size_t count = 1;
};

template<typename S, typename F>
struct getter_impl<S, F, static_cast<std::size_t>(std::tuple_size<typename sub_structures<S>::value_type>::value), std::enable_if_t<!can_apply_v<S, F>>> {
    static constexpr auto get(S, F) { return std::tuple<>{}; }
    static constexpr std::size_t count = 0;
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<can_apply_v<S, F>>> {
    static constexpr auto get(S s, F f) { return f(s); }
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<!can_apply_v<S, F> && (getter_impl<S, F>::count == 1)>> {
    static constexpr auto get(S s, F f) { return std::get<0>(getter_impl<S, F>::get(s, f)); }
};

template<typename S, typename F>
struct getter<S, F, std::enable_if_t<!can_apply_v<S, F> && (getter_impl<S, F>::count != 1)>> {
    static_assert(getter_impl<S, F>::count != 0, "getter has to be applicable");
    static_assert(!(getter_impl<S, F>::count > 1), "getter cannot be ambiguous");
};

struct transform_trait;
struct get_trait;

using default_trait = transform_trait;

template<typename F, typename = void>
struct func_trait {
    using type = default_trait;
};

template<typename F>
struct func_trait<F, std::enable_if_t<std::is_same<typename F::func_family, transform_trait>::value>> {
    using type = transform_trait;
};

template<typename F>
struct func_trait<F, std::enable_if_t<std::is_same<typename F::func_family, get_trait>::value>> {
    using type = get_trait;
};

template<typename F>
using func_trait_t = typename func_trait<F>::type;

template<typename F, typename = void>
struct pipe_decider;

template<typename F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, transform_trait>::value>> {
    template<typename S>
    static constexpr auto operate(S s, F f) { return fmapper<S, F>::fmap(s, f);  }
};

template<typename F>
struct pipe_decider<F, std::enable_if_t<std::is_same<func_trait_t<F>, get_trait>::value>> {
    template<typename S>
    static constexpr auto operate(S s, F f) { return getter<S, F>::get(s, f);  }
};

template<typename S, typename F>
constexpr auto operator%(S s, F f) {
    return pipe_decider<F>::template operate<S>(s, f);
}

template<typename S, typename... Fs>
constexpr auto pipe(S s, Fs... funcs);

template<typename S, typename... Fs>
struct piper;

template<typename S, typename F, typename... Fs>
struct piper<S, F, Fs...> {
    static constexpr auto pipe(S s, F func, Fs... funcs) {
        auto s1 = s % func;
        return piper<decltype(s1), Fs...>::pipe(s1, funcs...);
    }
};

template<typename S, typename F>
struct piper<S, F> {
    static constexpr auto pipe(S s, F func) {
        return s % func;
    }
};

template<typename S, typename... Fs>
constexpr auto pipe(S s, Fs... funcs) {
    return piper<S, Fs...>::pipe(s, funcs...);
}

}

#endif  // NOARR_CORE_HPP