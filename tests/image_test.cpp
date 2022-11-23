#include <catch2/catch.hpp>

#include <array>
#include <iostream>
#include <tuple>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/wrapper.hpp>

TEST_CASE("Image", "[image]") {
	noarr::array<'x', 1920, noarr::array<'y', 1080, noarr::array<'p', 4, noarr::scalar<float>>>> g;
	auto grayscale = noarr::wrap(g);

	std::vector<char> my_blob_buffer(grayscale.get_size());
	char *my_blob = (char *)my_blob_buffer.data();

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(grayscale)>::value);
	}

	SECTION("value refs check") {
		std::size_t offset = grayscale.offset<'x', 'y', 'p'>(0, 0, 2);
		float& value_ref = *((float*)(my_blob + offset));
		value_ref = 0;
	}
}
