#include <iostream>

#include "noarr_structs.hpp"
#include "noarr_funcs.hpp"
#include "noarr_io.hpp"

using namespace noarr;

int main() {
    vector<'x', scalar<float>> v;

    array<'y', 20000, vector<'x', scalar<float>>> v2;
    tuple<'t', array<'x', 10, scalar<float>>, vector<'x', scalar<int>>> t;
    tuple<'t', array<'y', 20000, vector<'x', scalar<float>>>, vector<'x', array<'y', 20, scalar<int>>>> t2;

    static_assert(std::is_pod<decltype(v)>::value, "a struct has to be a podtype");
    static_assert(std::is_pod<decltype(v2)>::value, "a struct has to be a podtype");
    static_assert(std::is_pod<decltype(t)>::value, "a struct has to be a podtype");
    print_struct(std::cout, v) << " v;" << std::endl;
    print_struct(std::cout, v2) << " v2;" << std::endl;
    print_struct(std::cout, t) << " t;" << std::endl << std::endl;

    auto vs = v % resize<'x'>{10}; // transform
    static_assert(std::is_pod<decltype(vs)>::value, "a struct has to be a podtype");
    std::cout << "vs = v % resize<'x'>{10}: " << typeid(vs).name() << std::endl;
    std::cout << "vs.size(): " << vs.size() << std::endl;
    std::cout << "sizeof(vs): " << sizeof(vs) << std::endl << std::endl;

    auto vs2 = v2 % resize<'x'>{20}; // transform
    static_assert(std::is_pod<decltype(vs2)>::value, "a struct has to be a podtype");
    std::cout << "vs2 = v % resize<'x'>{10}: " << typeid(vs2).name() << std::endl;
    std::cout << "vs2.size(): " << vs2.size() << std::endl;
    std::cout << "sizeof(vs2): " << sizeof(vs2) << std::endl;
    std::cout << "vs2 % fix<'x'>{5}:" << typeid(vs2 % fix<'x'>{5}).name() << std::endl;
    std::cout << "vs2 % fix<'x'>{5} % fix<'y'>{5}:" << typeid(vs2 % fix<'x'>{5} % fix<'y'>{5}).name() << std::endl;
    std::cout << "vs2 % fixs<'x', 'y'>{5, 5}:" << typeid(vs2 % fixs<'x', 'y'>{5, 5}).name() << std::endl;
    std::cout << "vs2 % fixs<'x', 'y'>{5, 5} % offset{}:" << vs2 % fixs<'x', 'y'>{5, 5} % offset{} << std::endl;
    std::cout << "vs2 % fixs<'x', 'y'>{5, 5} % offset{}:" << (vs2 % fixs<'y', 'x'>{5, 5} % offset()) << std::endl;
    static_assert(std::is_pod<decltype(vs2 % fixs<'y', 'x'>{5, 5})>::value, "a struct has to be a podtype");
    static_assert(std::is_pod<decltype(fixs<'y', 'x'>{5, 5})>::value, "a struct has to be a podtype");
    std::cout << "vs2 % get_offset<'y'>{5}:" << (vs2 % get_offset<'y'>{5}) << std::endl << std::endl;

    auto vs3 = v2 % cresize<'x', 10>{}; // transform
    static_assert(std::is_pod<decltype(vs3)>::value, "a struct has to be a podtype");
    std::cout << "vs3 = v % cresize<'x', 10>{}: " << typeid(vs3).name() << std::endl;
    std::cout << "vs3.size(): " << vs3.size() << std::endl;
    std::cout << "sizeof(vs3): " << sizeof(vs3) << std::endl << std::endl;

    volatile std::size_t l = 20;
    std::cout << "choose l... ";
    auto vs4 = pipe(v2, cresize<'y', 10>{}, resize<'x'>{l}); // transform
    std::cout << "vs4 = pipe(v2, cresize<'y', 10>{}, resize<'x'>{l}): " << typeid(vs4).name() << std::endl;
    std::cout << "vs4.size(): " << vs4.size() << std::endl;
    std::cout << "sizeof(vs4): " << sizeof(vs4) << std::endl << std::endl;
    static_assert(std::is_pod<decltype(vs4)>::value, "a struct has to be a podtype");

    std::cout << "sizeof(t): " << sizeof(t) << std::endl;
    auto ts = t % resize<'x'>(20);
    print_struct(std::cout, ts) << " ts;" << std::endl;
    std::cout << "sizeof(ts): " << sizeof(ts) << std::endl;
    std::cout << "ts.size(): " << ts.size() << std::endl;
    static_assert(std::is_pod<decltype(ts)>::value, "a struct has to be a podtype");

    print_struct(std::cout, t2 % reassemble<'x', 'y'>{}) << " t2';" << std::endl;
    print_struct(std::cout, t2 % resize<'x'>(10) % reassemble<'y', 'x'>{}) << " t2'';" << std::endl;
    print_struct(std::cout, t2 % reassemble<'x', 'x'>{}) << " t2;" << std::endl;
}
