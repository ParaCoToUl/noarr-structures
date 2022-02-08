/**
 * @file compile_test.cpp
 * @brief This file contains the code snippets from the root README.md ensuring they are correct
 * 
 */

#include <catch2/catch.hpp>

#include <algorithm>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/dimension_map.hpp"

TEST_CASE("Dimension map", "[utility]") {
	noarr::helpers::dimension_map<noarr::helpers::index_pair<'x', std::size_t>, noarr::helpers::index_pair<'y', std::uint16_t *>, noarr::helpers::index_pair<'z', std::uint8_t>> map(5, (uint16_t *)std::uintptr_t(10), 'b');

	REQUIRE(sizeof(map) <= std::max({sizeof(std::size_t), sizeof(std::uint16_t *), sizeof(std::uint8_t)}) * 3);
	REQUIRE(alignof(decltype(map)) >= std::max({alignof(std::size_t), alignof(std::uint16_t *), alignof(std::uint8_t)}));

	REQUIRE(sizeof(map.get<'x'>()) == sizeof(std::size_t));
	REQUIRE(sizeof(map.get<'y'>()) == sizeof(std::uint16_t *));
	REQUIRE(sizeof(map.get<'z'>()) == sizeof(std::uint8_t));

	REQUIRE(map.get<'x'>() == 5);
	REQUIRE((std::uintptr_t)map.get<'y'>() == 10);
	REQUIRE(map.get<'z'>() == 'b');

	map.get<'x'>() += 5;
	map.get<'y'>() += 5;
	map.get<'z'>() += 255;

	REQUIRE(map.get<'x'>() == 10);
	REQUIRE((std::uintptr_t)map.get<'y'>() == 10 + 5 * sizeof(std::uint16_t));
	REQUIRE(map.get<'z'>() == 'a');
}
