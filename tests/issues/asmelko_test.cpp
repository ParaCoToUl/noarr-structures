#include <catch2/catch.hpp>

#include <noarr/structures_extended.hpp>

template <typename T>
using matrix_rows = noarr::vector<'m', noarr::vector<'n', noarr::scalar<T>>>;

TEST_CASE("ASmelko0", "[user issue]") {
    size_t size = 0;

    auto rows_struct = matrix_rows<double>() | noarr::set_length<'n'>(size) | noarr::set_length<'m'>(size);

    const auto bag = noarr::make_bag(rows_struct);
    volatile auto x = bag.template at<'m','n'>(0,0);
    x = x;
}
