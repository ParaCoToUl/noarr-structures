#include <catch2/catch.hpp>

#include <noarr/structures.hpp>
#include <noarr/structures/extra/shortcuts.hpp>
#include "noarr_test_defs.hpp"

TEST_CASE("State shortcuts", "[state]") {
	auto s1 = noarr::make_state<noarr::length_in<'x'>, noarr::length_in<'y'>>(10, 20);

	REQUIRE(sizeof(s1) == 2*sizeof(std::size_t));
	REQUIRE(noarr_test::type_is_simple(s1));

	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s1), noarr::length_in<'x'>>, std::size_t>);
	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s1), noarr::length_in<'y'>>, std::size_t>);

	REQUIRE(s1.get<noarr::length_in<'x'>>() == 10);
	REQUIRE(s1.get<noarr::length_in<'y'>>() == 20);

	auto s2 = s1 & noarr::idx<'x', 'y'>(05, 15);

	REQUIRE(sizeof(s2) == 4*sizeof(std::size_t));
	REQUIRE(noarr_test::type_is_simple(s2));

	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s2), noarr::length_in<'x'>>, std::size_t>);
	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s2), noarr::length_in<'y'>>, std::size_t>);
	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s2), noarr::index_in<'x'>>, std::size_t>);
	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s2), noarr::index_in<'y'>>, std::size_t>);

	REQUIRE(s2.get<noarr::length_in<'x'>>() == 10);
	REQUIRE(s2.get<noarr::length_in<'y'>>() == 20);
	REQUIRE(s2.get<noarr::index_in<'x'>>() == 05);
	REQUIRE(s2.get<noarr::index_in<'y'>>() == 15);

	auto [s2x, s2y] = noarr::get_indices<'x', 'y'>(s2);

	REQUIRE(noarr::get_index<'x'>(s2) == 05);
	REQUIRE(noarr::get_index<'y'>(s2) == 15);
	REQUIRE(s2x == 05);
	REQUIRE(s2y == 15);
}

TEST_CASE("State shortcuts lit", "[state]") {
	auto s1 = noarr::make_state<noarr::length_in<'x'>, noarr::length_in<'y'>>(noarr::lit<10>, noarr::lit<20>);

	REQUIRE(std::is_empty_v<decltype(s1)>);
	REQUIRE(noarr_test::type_is_simple(s1));

	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s1), noarr::length_in<'x'>>, std::integral_constant<std::size_t, 10>>);
	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s1), noarr::length_in<'y'>>, std::integral_constant<std::size_t, 20>>);

	REQUIRE(decltype(s1.get<noarr::length_in<'x'>>())::value == 10);
	REQUIRE(decltype(s1.get<noarr::length_in<'y'>>())::value == 20);

	auto s2 = s1 & noarr::idx<'x', 'y'>(noarr::lit<05>, noarr::lit<15>);

	REQUIRE(sizeof(s2) == 1);
	REQUIRE(noarr_test::type_is_simple(s2));

	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s2), noarr::length_in<'x'>>, std::integral_constant<std::size_t, 10>>);
	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s2), noarr::length_in<'y'>>, std::integral_constant<std::size_t, 20>>);
	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s2), noarr::index_in<'x'>>, std::integral_constant<std::size_t, 05>>);
	REQUIRE(std::is_same_v<noarr::state_get_t<decltype(s2), noarr::index_in<'y'>>, std::integral_constant<std::size_t, 15>>);

	REQUIRE(decltype(s2.get<noarr::length_in<'x'>>())::value == 10);
	REQUIRE(decltype(s2.get<noarr::length_in<'y'>>())::value == 20);
	REQUIRE(decltype(s2.get<noarr::index_in<'x'>>())::value == 05);
	REQUIRE(decltype(s2.get<noarr::index_in<'y'>>())::value == 15);

	auto [s2x, s2y] = noarr::get_indices<'x', 'y'>(s2);

	REQUIRE(decltype(noarr::get_index<'x'>(s2))::value == 05);
	REQUIRE(decltype(noarr::get_index<'y'>(s2))::value == 15);
	REQUIRE(decltype(s2x)::value == 05);
	REQUIRE(decltype(s2y)::value == 15);
}
