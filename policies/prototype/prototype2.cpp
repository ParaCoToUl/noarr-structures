#include <cassert>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <tuple>

namespace arrr {

template<class... T>
using void_t = void;

template<class T>
using remove_cvref = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

struct empty_struct_t { private: const char _value[0] = {}; };

template<class T>
struct is_array {
    static constexpr bool value = false;
};

template<class T, std::size_t N>
struct is_array<T[N]> {
    static constexpr bool value = true;
};

template<class T>
struct array_length;

template<class T, std::size_t N>
struct array_length<T[N]> {
    static constexpr std::size_t value = N;
};

template<typename T, typename = void>
struct sub_structures {
    explicit sub_structures(T t) {}
    using value_type = std::tuple<>;
    static constexpr std::tuple<> value = std::tuple<>{};
};

// TODO: check if tuple
template<typename T>
struct sub_structures<T, void_t<decltype(T::sub_structures)>> {
    explicit sub_structures(T t) : value {t.sub_structures} {}
    using value_type = remove_cvref<decltype(T::sub_structures)>;
    const decltype(T::sub_structures) value;
};

template<typename T, typename = void>
struct dims { 
    static constexpr char value[] = {};
};

template<typename T>
struct dims<T, typename std::enable_if<std::is_array<remove_cvref<decltype(T::dims)>>::value>::type> {
    static constexpr auto &&value = T::dims;
};

template<typename T>
struct ndims {
    static constexpr std::size_t value = array_length<remove_cvref<decltype(dims<T>::value)>>::value;
};

template<typename S, typename F>
constexpr auto operator%(S s, F f);

template<typename S, typename F, typename = void>
struct fmapper;

template<typename S, typename F, typename = void>
struct can_apply;

struct fmap_traversor;

template<std::size_t I, std::size_t J = 0>
struct fmap_traversor_impl;

template<std::size_t I, std::size_t J>
struct fmap_traversor_impl {
    friend struct fmap_traversor;
    template<std::size_t, std::size_t>
    friend struct fmap_traversor_impl;

private:
    template<typename S, typename F, std::size_t ...Is>
    static constexpr auto sub_fmap_build(S s, F f) {
        return fmap_traversor_impl<I, J + 1>::template sub_fmap_build<S, F, J, Is...>(s, f);
    }
};

template<std::size_t I>
struct fmap_traversor_impl<I, I> {
    friend struct fmap_traversor;
    template<std::size_t, std::size_t>
    friend struct fmap_traversor_impl;

private:
    template<typename S, typename F, std::size_t ...Is>
    static constexpr auto sub_fmap_explicit(S s, F f) {
        return S::construct((std::get<I - 1 - Is>(sub_structures<S>{s}.value) % f) ...);
    }

    template<typename S, typename F, std::size_t ...Is>
    static constexpr auto sub_fmap_build(S s, F f) {
        return sub_fmap_explicit<S, F, Is...>(s, f);
    }
};

template<>
struct fmap_traversor_impl<0, 0> {
private:
    template<typename S, typename F>
    static constexpr auto sub_fmap_explicit(S s, F f) {
        return s;
    }
public:
    template<typename S, typename F>
    static constexpr auto sub_fmap_build(S s, F f) {
        return sub_fmap_explicit<S, F>(s, f);
    }
};

struct fmap_traversor {
    template<typename S, typename F>
    static constexpr auto sub_fmap(S s, F f) {
        return fmap_traversor_impl<std::tuple_size<typename sub_structures<S>::value_type>::value>::template sub_fmap_build<S, F>(s, f);
    }
};

template<typename S, typename F, typename>
struct can_apply { static constexpr bool value = false; };

template<typename S, typename F>
struct can_apply<S, F, void_t<decltype(F{}(S{}))>> { static constexpr bool value = true; };

template<typename S, typename F>
constexpr bool can_apply_v = can_apply<S, F, void>::value;

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<!can_apply_v<S, F>, void_t<decltype(fmap_traversor::sub_fmap(S{}, F{}))>>>  {
    static constexpr auto fmap(S s, F f) {
        return fmap_traversor::sub_fmap(s, f);
    }
};

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<can_apply_v<S, F>>> {
    static constexpr auto fmap(S s, F f) {
        return f(s);
    }

};

template<typename S, typename F>
constexpr auto operator%(S s, F f) {
    return fmapper<S, F>::fmap(s, f);
}

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
struct getter_impl<S, F, (std::size_t)std::tuple_size<typename sub_structures<S>::value_type>::value, std::enable_if_t<!can_apply_v<S, F>>> {
    static constexpr auto get(S s, F f) { return std::tuple<>{}; }
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

template<typename S, typename F>
constexpr auto operator/(S s, F f) {
    return getter<S, F>::get(s, f);
}

}

using namespace arrr;

struct A {

};

struct B {
    template<typename A, typename C>
    static constexpr B construct(A a, C b) {
        std::cout << "constructing a B from " << typeid(A).name() << " and " << typeid(C).name() << std::endl;
        return B{};
    }
    static constexpr std::tuple<A, A> sub_structures = {{}, {}};
};

struct D {

};

struct E {
    static constexpr std::tuple<A, D> sub_structures = {{}, {}};
};

struct X {
    int operator()(A t) { return 0; }
};

struct Y {
    float operator()(D t) { return 0; }
};

struct Z {
    char operator()(B t) { return 0; }
};

int main() {
    A a;
    B b;
    E e;
    b % X{};
    std::cout << typeid(b % X{}).name() << std::endl;
    e / X{};
    std::cout << typeid(e / X{}).name() << std::endl;
    b % Y{};
    std::cout << typeid(b % Y{}).name() << std::endl;
    e / Y{};
    std::cout << typeid(e / Y{}).name() << std::endl;
    b % Z{};
    std::cout << typeid(b % Z{}).name() << std::endl;
    // e / Z{};
    // std::cout << typeid(e / Z{}).name() << std::endl;
    // b / X{};
    // std::cout << typeid(b / X{}).name() << std::endl;
}