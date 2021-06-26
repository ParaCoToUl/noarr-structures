// pybind11_wrapper.cpp
#include <pybind11/pybind11.h>
#include <algorithm.cpp>

PYBIND11_MODULE(foo, m) {
    m.doc() = "pybind11 example plugin"; // Optional module docstring
    m.def("example_function", &foo::example_function, "A function that says stuff");
}
