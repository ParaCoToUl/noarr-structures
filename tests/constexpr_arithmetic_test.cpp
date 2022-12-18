#include <catch2/catch.hpp>

#include <noarr/structures_extended.hpp>

using namespace noarr;

TEST_CASE("Array constexpr arithmetic", "[cearithm]") {
	auto s = scalar<int>() ^ array<'x', 10>() ^ array<'y', 20>();
	REQUIRE(decltype(s | get_size())::value == 10 * 20 * sizeof(int));
	REQUIRE(decltype(s | get_length<'x'>())::value == 10);
	REQUIRE(decltype(s | get_length<'y'>())::value == 20);
	REQUIRE(decltype(s | offset<'y', 'x'>(idx<15>, idx<5>))::value == (15*10 + 5) * sizeof(int));
}

TEST_CASE("Vector constexpr arithmetic", "[cearithm]") {
	auto s = scalar<int>() ^ vector<'x'>() ^ vector<'y'>() ^ set_length<'x'>(idx<10>) ^ set_length<'y'>(idx<20>);
	REQUIRE(decltype(s | get_size())::value == 10 * 20 * sizeof(int));
	REQUIRE(decltype(s | get_length<'x'>())::value == 10);
	REQUIRE(decltype(s | get_length<'y'>())::value == 20);
	REQUIRE(decltype(s | offset<'y', 'x'>(idx<15>, idx<5>))::value == (15*10 + 5) * sizeof(int));
}

TEST_CASE("Block split constexpr arithmetic", "[cearithm]") {
	auto s = scalar<int>() ^ array<'i', 200>() ^ into_blocks<'i', 'y', 'x'>(idx<10>);
	REQUIRE(decltype(s | get_size())::value == 200 * sizeof(int));
	REQUIRE(decltype(s | get_length<'x'>())::value == 10);
	REQUIRE(decltype(s | get_length<'y'>())::value == 200/10);
	REQUIRE(decltype(s | offset<'y', 'x'>(idx<15>, idx<5>))::value == (15*10 + 5) * sizeof(int));
}

TEST_CASE("Block merge constexpr arithmetic", "[cearithm]") {
	auto s = scalar<int>() ^ array<'x', 10>() ^ array<'y', 20>() ^ merge_blocks<'y', 'x', 'i'>();
	REQUIRE(decltype(s | get_size())::value == 10 * 20 * sizeof(int));
	REQUIRE(decltype(s | get_length<'i'>())::value == 10 * 20);
	REQUIRE(decltype(s | offset<'i'>(idx<155>))::value == 155 * sizeof(int));
}

TEST_CASE("Shift outer constexpr arithmetic", "[cearithm]") {
	auto s = scalar<int>() ^ array<'x', 10>() ^ array<'y', 20>() ^ shift<'y'>(idx<3>);
	REQUIRE(decltype(s | get_size())::value == 10 * 20 * sizeof(int));
	REQUIRE(decltype(s | get_length<'x'>())::value == 10);
	REQUIRE(decltype(s | get_length<'y'>())::value == 17);
	REQUIRE(decltype(s | offset<'y', 'x'>(idx<12>, idx<5>))::value == (15*10 + 5) * sizeof(int));
}

TEST_CASE("Shift inner constexpr arithmetic", "[cearithm]") {
	auto s = scalar<int>() ^ array<'x', 10>() ^ array<'y', 20>() ^ shift<'x'>(idx<3>);
	REQUIRE(decltype(s | get_size())::value == 10 * 20 * sizeof(int));
	REQUIRE(decltype(s | get_length<'x'>())::value == 7);
	REQUIRE(decltype(s | get_length<'y'>())::value == 20);
	REQUIRE(decltype(s | offset<'y', 'x'>(idx<15>, idx<2>))::value == (15*10 + 5) * sizeof(int));
}
