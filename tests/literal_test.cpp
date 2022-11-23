#include <catch2/catch.hpp>

#include <noarr/structures.hpp>
#include <noarr/structures/structs/setters.hpp>

using namespace noarr;

TEST_CASE("Literality test", "[low-lvl]") {
	constexpr auto test_array = array<'x', 100, scalar<int>>();
	constexpr auto test_array_size = test_array | get_size();

	constexpr auto test_vector = vector<'x', scalar<int>>();
	constexpr auto test_sized_vector = test_vector ^ set_length<'x'>(100);
	constexpr auto test_sized_vector_size = test_sized_vector | get_size();

	constexpr auto test_tuple = tuple<'t', scalar<int>, array<'x', 50, scalar<int>>, array<'y', 50, scalar<int>>>();
	constexpr auto test_tuple_size = test_tuple | get_size();

	constexpr auto test_sfixed_dim = vector<'x', scalar<int>>() ^ set_length<'x'>(0) ^ fix<'x'>(idx<10>);
	constexpr auto test_sfixed_dim_size = test_sfixed_dim | get_size();

	constexpr auto test_fixed_dim = vector<'x', scalar<int>>() ^ set_length<'x'>(0) ^ fix<'x'>(10);
	constexpr auto test_fixed_dim_size = test_fixed_dim | get_size();

	REQUIRE(std::integral_constant<std::size_t, (test_array_size + test_sized_vector_size + test_tuple_size + test_sfixed_dim_size + test_fixed_dim_size)>::value == 1204UL);
}
