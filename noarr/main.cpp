#include <iostream>

#include "noarr_types.hpp"

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
    static constexpr std::tuple<T1, T2> sub_structures = {{}, {}};
    template<typename A, typename C>
    static constexpr F<A, C> construct(A a, C b) {
        std::cout << "constructing a F<" << typeid(A).name() << "," << typeid(C).name() << "> from " << typeid(A).name() << " and " << typeid(C).name() << std::endl;
        return F<A, C>{};
    }
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