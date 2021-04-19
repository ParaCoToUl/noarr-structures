#include <iostream>

#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"

using namespace noarr;

int main() {

    vector<'x', scalar<float>> v;
    array<'y', 20000, vector<'x', scalar<float>>> v2;
    tuple<'t', array<'x', 10, scalar<float>>, vector<'x', scalar<int>>> t;
    tuple<'t', array<'y', 20000, vector<'x', scalar<float>>>, vector<'x', array<'y', 20, scalar<int>>>> t2;

    tuple<'t', scalar<int>, scalar<int>> t3;

    std::cout << "pipe(t3, fix<'t'>(1_idx), offset()): " << pipe(t3, fix<'t'>(1_idx), offset()) << std::endl;

    static_assert(!is_cube<decltype(v)>::value, "t must not be a cube");
    static_assert(!is_cube<decltype(t)>::value, "t must not be a cube");
    static_assert(!is_cube<decltype(t2)>::value, "t2 must not be a cube");

    static_assert(std::is_pod<decltype(v)>::value, "a struct has to be a podtype");
    static_assert(std::is_pod<decltype(v2)>::value, "a struct has to be a podtype");
    static_assert(std::is_pod<decltype(t)>::value, "a struct has to be a podtype");
    print_struct(std::cout, v) << " v;" << std::endl;
    print_struct(std::cout, v2) << " v2;" << std::endl;
    print_struct(std::cout, t) << " t;" << std::endl << std::endl;

    auto vs = v | set_length<'x'>(10); // transform
    static_assert(is_cube<decltype(vs)>::value, "vs has to be a cube");
    static_assert(std::is_pod<decltype(vs)>::value, "a struct has to be a podtype");
    std::cout << "vs = v | set_length<'x'>(10): " << typeid(vs).name() << std::endl;
    std::cout << "vs.size(): " << vs.size() << std::endl;
    std::cout << "vs | get_length<'x>(): " << (vs | get_length<'x'>()) << std::endl;
    std::cout << "sizeof(vs): " << sizeof(vs) << std::endl << std::endl;

    auto vs2 = v2 | set_length<'x'>(20); // transform
    static_assert(is_cube<decltype(vs2)>::value, "vs2 has to be a cube");
    static_assert(std::is_pod<decltype(vs2)>::value, "a struct has to be a podtype");
    std::cout << "vs2 = v | set_length<'x'>(10): " << typeid(vs2).name() << std::endl;
    std::cout << "vs2.size(): " << vs2.size() << std::endl;
    std::cout << "sizeof(vs2): " << sizeof(vs2) << std::endl;
    std::cout << "vs2 | fix<'x'>(5):" << typeid(vs2 | fix<'x'>(5)).name() << std::endl;
    std::cout << "vs2 | fix<'x'>(5) | fix<'y'>(6):" << typeid(vs2 | fix<'x'>(5) | fix<'y'>(6)).name() << std::endl;
    std::cout << "vs2 | fix<'x', 'y'>(5, 6):" << typeid(vs2 | fix<'x', 'y'>(5, 6)).name() << std::endl;
    std::cout << "vs2 | fix<'x', 'y'>(5, 6) | offset():" << (vs2 | fix<'x', 'y'>(5, 6) | offset()) << std::endl;
    std::cout << "vs2 | fix<'y', 'x'>(6, 5) | offset():" << (vs2 | fix<'y', 'x'>(6, 5) | offset()) << std::endl;
    std::cout << "vs2 | offset<'x', 'y'>(5, 6):" << (vs2 | offset<'x', 'y'>(5, 6)) << std::endl;
    std::cout << "vs2 | offset<'y', 'x'>(6, 5):" << (vs2 | offset<'y', 'x'>(6, 5)) << std::endl;
    std::cout << "vs2 | offset<'y', 'x'>(5, 6):" << (vs2 | offset<'y', 'x'>(5, 6)) << std::endl;

    std::cout << "vs2 | fix<'y', 'x'>(6, 5) | get_at((char *)nullptr): " << typeid(vs2 | fix<'y', 'x'>(6, 5) | get_at((char *)nullptr)).name() << std::endl;
    std::cout << "vs2 | get_at<'y', 'x'>((char *)nullptr, 6, 5): " << typeid(vs2 | get_at<'y', 'x'>((char *)nullptr, 6, 5)).name() << std::endl;

    static_assert(std::is_pod<decltype(vs2 | fix<'y', 'x'>(5, 5))>::value, "a struct has to be a podtype");
    static_assert(std::is_pod<decltype(fix<'y', 'x'>(5, 5))>::value, "a struct has to be a podtype");
    std::cout << "vs2 | get_offset<'y'>(5):" << (vs2 | get_offset<'y'>(5)) << std::endl << std::endl;
    static_assert(is_point<decltype(vs2 | fix<'y', 'x'>(5, 5))>::value, "`vs2 | fix<'y', 'x'>(5, 5)` has to be a point");

    auto vs3 = v2 | cresize<'x', 10>(); // transform
    static_assert(is_cube<decltype(vs3)>::value, "vs3 has to be a cube");
    static_assert(std::is_pod<decltype(vs3)>::value, "a struct has to be a podtype");
    std::cout << "vs3 = v | cresize<'x', 10>(): " << typeid(vs3).name() << std::endl;
    std::cout << "vs3.size(): " << vs3.size() << std::endl;
    std::cout << "sizeof(vs3): " << sizeof(vs3) << std::endl << std::endl;

    volatile std::size_t l = 20;
    std::cout << "choose l... ";

    auto vs4 = pipe(v2, cresize<'y', 10>(), set_length<'x'>(l)); // transform
    static_assert(is_cube<decltype(vs4)>::value, "vs4 has to be a cube");
    static_assert(std::is_pod<decltype(vs4)>::value, "vs4 has to be a podtype");
    std::cout << "vs4 = pipe(v2, cresize<'y', 10>(), set_length<'x'>(l)): " << typeid(vs4).name() << std::endl;
    std::cout << "vs4.size(): " << vs4.size() << std::endl;
    std::cout << "vs4 | get_length<'y'>(): " << (vs4 | get_length<'y'>()) << std::endl;
    std::cout << "sizeof(vs4): " << sizeof(vs4) << std::endl << std::endl;

    std::cout << "sizeof(t): " << sizeof(t) << std::endl;

    auto ts = t | set_length<'x'>(20);
    static_assert(!is_cube<decltype(ts)>::value, "ts must not be a cube");
    static_assert(std::is_pod<decltype(ts)>::value, "ts has to be a podtype");
    print_struct(std::cout, ts) << " ts;" << std::endl;
    std::cout << "sizeof(ts): " << sizeof(ts) << std::endl;
    std::cout << "ts.size(): " << ts.size() << std::endl;
    // std::cout << "ts | get_length<'x'>(): " << (ts | fix<'t'>(0_idx) | get_length<'x'>()) << std::endl;

    std::cout << "ts | fix<'t', 'x'>(0_idx, 5) | offset(): " << (ts | fix<'t', 'x'>(0_idx, 5) | offset()) << std::endl;
    std::cout << "ts | fix<'t', 'x'>(1_idx, 5) | offset(): " << (ts | fix<'t', 'x'>(1_idx, 5) | offset()) << std::endl;
    std::cout << "ts | fix<'x', 't'>(5, 1_idx) | offset(): " << (ts | fix<'x', 't'>(5, 1_idx) | offset()) << std::endl;

    static_assert(std::is_literal_type<decltype(fix<'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'>(1_idx, 1, 1_idx, 1, 1_idx, 1, 1_idx, 1))>::value, "it has to be a pod");

    print_struct(std::cout, t2 | reassemble<'x', 'y'>()) << " t2';" << std::endl;
    print_struct(std::cout, t2 | set_length<'x'>(10) | reassemble<'y', 'x'>()) << " t2'';" << std::endl;
    print_struct(std::cout, t2 | reassemble<'x', 'x'>()) << " t2;" << std::endl;
}
