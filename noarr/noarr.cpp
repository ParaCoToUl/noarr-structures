#include <iostream>
#include <type_traits>
#include <cassert>
#include <cstddef>
#include <tuple>

namespace noarr {

template<class... T>
using void_t = void;

template<class T>
using remove_cvref = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

struct empty_struct_t {
    constexpr empty_struct_t() : _value{} {}
    const char _value[0];
};

template<class T>
struct is_array {
    using value_type = bool;
    static constexpr value_type value = false;
};

template<class T, std::size_t N>
struct is_array<T[N]> {
    using value_type = bool;
    static constexpr value_type value = true;
};

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
    explicit sub_structures(T t) {}
    using value_type = std::tuple<>;
    static constexpr std::tuple<> value = std::tuple<>{};
};

template<class T, char DIM, typename = void>
struct dims_have;

template<char NEEDLE>
struct dims_have<dims_impl<>, NEEDLE> {
    using value_type = bool;
    static constexpr value_type value = false;
};

template<char NEEDLE, char... HAY_TAIL>
struct dims_have<dims_impl<NEEDLE, HAY_TAIL...>, NEEDLE> {
    using value_type = bool;
    static constexpr value_type value = true;
};

template<char NEEDLE, char HAY_HEAD, char... HAY_TAIL>
struct dims_have<dims_impl<HAY_HEAD, HAY_TAIL...>, NEEDLE, std::enable_if_t<HAY_HEAD != NEEDLE>> {
    using value_type = bool;
    static constexpr value_type value = dims_have<dims_impl<HAY_TAIL...>, NEEDLE>::value;
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
    static constexpr dims_impl<> value = {};
};

// TODO: check if dims_impl
template<typename T>
struct dims<T, void_t<decltype(T::dims)>> {
    static constexpr auto value = T::dims;
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
        return s.construct((std::get<I - 1 - Is>(sub_structures<S>{s}.value) % f) ...);
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
struct can_apply<S, F, void_t<decltype(std::declval<F>()(std::declval<S>()))>> { static constexpr bool value = true; };

template<typename S, typename F>
constexpr bool can_apply_v = can_apply<S, F, void>::value;

template<typename S, typename F>
struct fmapper<S, F, std::enable_if_t<!can_apply_v<S, F>, void_t<decltype(fmap_traversor::sub_fmap(std::declval<S>(), std::declval<F>()))>>>  {
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

template <typename T>
using is_empty = typename std::is_base_of<empty_struct_t, T>;

template<typename T>
struct scalar {
    std::tuple<> sub_structures;
    static constexpr dims_impl<> dims = {};
    constexpr scalar() : sub_structures{} {}
    static constexpr scalar<T> construct() {
        return {};
    }
    static constexpr std::size_t size() { return sizeof(T); }
};

template<char DIM, typename T>
struct vector {
    std::tuple<T> sub_structures;
    static constexpr dims_impl<DIM> dims = {};
    constexpr vector() : sub_structures{{}} {}
    constexpr vector(T sub_structure) : sub_structures{std::make_tuple(sub_structure)} {}
    template<typename T2>
    static constexpr vector<DIM, T2> construct(T2 sub_structure) {
        return {sub_structure};
    }
};

template<char DIM, std::size_t L, typename T>
struct array {
    std::tuple<T> sub_structures;
    static constexpr std::size_t length = L;
    static constexpr dims_impl<DIM> dims = {};
    constexpr array() : sub_structures{{}} {}
    constexpr array(T sub_structure) : sub_structures{std::make_tuple(sub_structure)} {}
    template<typename T2>
    static constexpr array<DIM, L, T2> construct(T2 sub_structure) {
        return {sub_structure};
    }

    constexpr std::size_t size() { return std::get<0>(sub_structures).size() * L; }
    constexpr std::size_t offset(std::size_t i) { return std::get<0>(sub_structures).size() * i; }
};

template<char DIM, typename T>
struct sized_vector : vector<DIM, T> {
    std::size_t length;
    using get_t = T;
    using vector<DIM, T>::dims;
    using vector<DIM, T>::sub_structures;
    constexpr sized_vector(T sub_structure, std::size_t length) : vector<DIM, T>{sub_structure}, length{length} {}
    template<typename T2>
    constexpr sized_vector<DIM, T2> construct(T2 sub_structure) const {
        return {sub_structure, length};
    }

    constexpr std::size_t size() { return std::get<0>(sub_structures).size() * length; }
    constexpr std::size_t offset(std::size_t i) { return std::get<0>(sub_structures).size() * i; }
};

template<char DIM>
struct resize {
    std::size_t length;
    constexpr resize(std::size_t length) : length{length} {}
    template<typename T>
    constexpr sized_vector<DIM, T> operator()(vector<DIM, T> v) const {
        return {std::get<0>(v.sub_structures), length};
    }
    template<typename T>
    constexpr sized_vector<DIM, T> operator()(sized_vector<DIM, T> v) const {
        return {std::get<0>(v.sub_structures), length};
    }
};

}

using namespace noarr;

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

template<typename T1, typename T2>
struct F {
    template<typename A, typename C>
    static constexpr F<A, C> construct(A a, C b) {
        std::cout << "constructing a F<" << typeid(A).name() << "," << typeid(C).name() << "> from " << typeid(A).name() << " and " << typeid(C).name() << std::endl;
        return F<A, C>{};
    }
    static constexpr std::tuple<T1, T2> sub_structures = {{}, {}};
};

struct X {
    int operator()(A t) { return 0; }
};

struct Y {
    float operator()(D t) { return 0; }
    using func_family = get_trait;
};

struct Z {
    char operator()(D t) { return 0; }
    using func_family = transform_trait;
};

int main() {
    B b;
    E e;
    F<A, D> f1;
    vector<'x', scalar<float>> v;
    array<'y', 20, vector<'x', scalar<float>>> v2;
    b % X{}; // transform
    std::cout << "b % X{}: " << typeid(b % X{}).name() << std::endl;
    e % Y{}; // get
    std::cout << "e % Y{}: " << typeid(e % Y{}).name() << std::endl;
    f1 % X{}; // transform
    std::cout << "f1 % X{}: " << typeid(f1 % X{}).name() << std::endl;
    f1 % Z{}; // transform
    std::cout << "f1 % Z{}: " << typeid(f1 % Z{}).name() << std::endl;
    pipe(f1, Z{}, X{}); // transform twice
    std::cout << "pipe(f1, Z{}, X{}): " << typeid(pipe(f1, Z{}, X{})).name() << std::endl;
    pipe(f1, X{}, Y{}); // transform and get
    std::cout << "pipe(f1, X{}, Y{}): " << typeid(pipe(f1, X{}, Y{})).name() << std::endl;
    
    auto vs = v % resize<'x'>{10}; // transform
    std::cout << "vs = v % resize<'x'>{10}: " << typeid(vs).name() << std::endl;
    std::cout << "vs.size(): " << vs.size() << std::endl;
    std::cout << "sizeof(vs) :( : " << sizeof(vs) << std::endl;
    
    auto vs2 = v2 % resize<'x'>{10}; // transform
    std::cout << "vs2 = v % resize<'x'>{10}: " << typeid(vs2).name() << std::endl;
    std::cout << "vs2.size(): " << vs2.size() << std::endl;
    std::cout << "sizeof(vs2) :( : " << sizeof(vs2) << std::endl;
}

// TODO: we want smarter tuples that understand emptiness (recursively and on each element separately)