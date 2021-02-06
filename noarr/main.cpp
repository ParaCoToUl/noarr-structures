#include <iostream>

#include "noarr_structs.hpp"
#include "noarr_funcs.hpp"
#include "noarr_io.hpp"

using namespace noarr;

int main() {
    vector<'x', scalar<float>> v;

    array<'y', 20000, vector<'x', scalar<float>>> v2;
    tuple<'t', array<'x', 10, scalar<float>>, array<'x', 10, scalar<int>>> t;

    static_assert(std::is_trivial<decltype(v)>::value, "a struct has to be trivial");
    static_assert(std::is_trivial<decltype(v2)>::value, "a struct has to be trivial");
    static_assert(std::is_trivial<decltype(t)>::value, "a struct has to be trivial");
    print_struct(std::cout, v) << " v;" << std::endl;
    print_struct(std::cout, v2) << " v2;" << std::endl;
    print_struct(std::cout, t) << " t;" << std::endl << std::endl;

    auto vs = v % resize<'x'>{10}; // transform
    static_assert(std::is_trivial<decltype(vs)>::value, "a struct has to be trivial");
    std::cout << "vs = v % resize<'x'>{10}: " << typeid(vs).name() << std::endl;
    std::cout << "vs.size(): " << vs.size() << std::endl;
    std::cout << "sizeof(vs): " << sizeof(vs) << std::endl << std::endl;

    auto vs2 = v2 % resize<'x'>{20}; // transform
    static_assert(std::is_trivial<decltype(vs2)>::value, "a struct has to be trivial");
    std::cout << "vs2 = v % resize<'x'>{10}: " << typeid(vs2).name() << std::endl;
    std::cout << "vs2.size(): " << vs2.size() << std::endl;
    std::cout << "sizeof(vs2): " << sizeof(vs2) << std::endl;
    std::cout << "vs2 % fix<'x'>{5}:" << typeid(vs2 % fix<'x'>{5}).name() << std::endl;
    std::cout << "vs2 % fix<'x'>{5} % fix<'y'>{5}:" << typeid(vs2 % fix<'x'>{5} % fix<'y'>{5}).name() << std::endl;
    std::cout << "vs2 % fixs<'x', 'y'>{5, 5}:" << typeid(vs2 % fixs<'x', 'y'>{5, 5}).name() << std::endl;
    std::cout << "vs2 % fixs<'x', 'y'>{5, 5} % offset{}:" << vs2 % fixs<'x', 'y'>{5, 5} % offset{} << std::endl;
    std::cout << "vs2 % fixs<'x', 'y'>{5, 5} % offset{}:" << (vs2 % fixs<'y', 'x'>{5, 5} % offset()) << std::endl;
    static_assert(std::is_trivial<decltype(vs2 % fixs<'y', 'x'>{5, 5})>::value, "a struct has to be trivial");
    std::cout << "vs2 % get_offset<'y'>{5}:" << (vs2 % get_offset<'y'>{5}) << std::endl << std::endl;

    auto vs3 = v2 % cresize<'x', 10>{}; // transform
    static_assert(std::is_trivial<decltype(vs3)>::value, "a struct has to be trivial");
    std::cout << "vs3 = v % cresize<'x', 10>{}: " << typeid(vs3).name() << std::endl;
    std::cout << "vs3.size(): " << vs3.size() << std::endl;
    std::cout << "sizeof(vs3): " << sizeof(vs3) << std::endl << std::endl;

    std::size_t l;
    std::cout << "choose l... ";
    std::cin >> l;
    auto vs4 = pipe(v2, cresize<'y', 10>{}, resize<'x'>{l}); // transform
    std::cout << "vs4 = pipe(v2, cresize<'y', 10>{}, resize<'x'>{l}): " << typeid(vs4).name() << std::endl;
    std::cout << "vs4.size(): " << vs4.size() << std::endl;
    std::cout << "sizeof(vs4): " << sizeof(vs4) << std::endl << std::endl;

    std::cout << "t.size(): " << t.size() << std::endl;
    std::cout << "sizeof(t): " << sizeof(t) << std::endl;
}
