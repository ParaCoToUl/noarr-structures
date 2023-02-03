#include <catch2/catch_test_macros.hpp>

#include <array>
#include <iostream>
#include <cstring>

#include <noarr/structures.hpp>
#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/extra/wrapper.hpp>
#include <noarr/structures/interop/bag.hpp>

using namespace noarr;

TEST_CASE("Wrapper traverser", "[to_struct]") {
	using at = noarr::array<'x', 2, noarr::array<'y', 3, noarr::scalar<int>>>;
	using bt = noarr::array<'y', 3, noarr::array<'z', 4, noarr::scalar<int>>>;

	auto a = at();
	auto b = bt();

	auto aw = a | wrap();
	auto bw = b | wrap();

	REQUIRE(std::is_empty_v<decltype(traverser(a, b))>);
	REQUIRE(std::is_same_v<decltype(traverser(a, b)), decltype(traverser(aw, bw))>);
}

TEST_CASE("Bag traverser", "[to_struct]") {
	using at = noarr::array<'x', 2, noarr::array<'y', 3, noarr::scalar<int>>>;
	using bt = noarr::array<'y', 3, noarr::array<'z', 4, noarr::scalar<int>>>;

	auto a = at();
	auto b = bt();

	auto aw = make_bag(a);
	auto bw = make_bag(b);

	REQUIRE(std::is_empty_v<decltype(traverser(a, b))>);
	REQUIRE(std::is_same_v<decltype(traverser(a, b)), decltype(traverser(aw, bw))>);
}

template<typename T1, typename T2>
static bool eq(const T1 &, const T2 &) { return false; }
template<typename T>
static bool eq(const T &t1, const T &t2) { return !std::memcmp(&t1, &t2, sizeof(T)); }

TEST_CASE("Vector wrapper traverser", "[to_struct]") {
	using at = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
	using bt = noarr::vector<'y', noarr::vector<'z', noarr::scalar<int>>>;

	auto a = at() ^ noarr::set_length<'x', 'y'>(2, 3);
	auto b = bt() ^ noarr::set_length<'y', 'z'>(3, 4);

	auto aw = a | wrap();
	auto bw = b | wrap();

	REQUIRE(eq(traverser(a, b), traverser(aw, bw)));
}

TEST_CASE("Vector bag traverser", "[to_struct]") {
	using at = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
	using bt = noarr::vector<'y', noarr::vector<'z', noarr::scalar<int>>>;

	auto a = at() ^ noarr::set_length<'x', 'y'>(2, 3);
	auto b = bt() ^ noarr::set_length<'y', 'z'>(3, 4);

	auto aw = make_bag(a);
	auto bw = make_bag(b);

	REQUIRE(eq(traverser(a, b), traverser(aw, bw)));
}
