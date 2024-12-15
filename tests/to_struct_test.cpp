#include <noarr_test/macros.hpp>
#include <noarr_test/defs.hpp>

#include <cstring>

#include <noarr/structures.hpp>
#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>

using namespace noarr;

TEST_CASE("Wrapper traverser", "[to_struct]") {
	using at = noarr::array_t<'x', 2, noarr::array_t<'y', 3, noarr::scalar<int>>>;
	using bt = noarr::array_t<'y', 3, noarr::array_t<'z', 4, noarr::scalar<int>>>;

	auto a = at();
	auto b = bt();

	auto aw = make_bag(a, (char *)nullptr);
	auto bw = make_bag(b, (char *)nullptr);

	STATIC_REQUIRE(std::is_same_v<decltype(aw), decltype(bag(a, (char *)nullptr))>);
	STATIC_REQUIRE(std::is_same_v<decltype(bw), decltype(bag(b, (char *)nullptr))>);

	REQUIRE(std::is_empty_v<decltype(traverser(a, b))>);
	STATIC_REQUIRE(std::is_same_v<decltype(traverser(a, b)), decltype(traverser(aw, bw))>);

	REQUIRE(noarr_test::equal_data(traverser(a, b), traverser(aw, bw)));
}

TEST_CASE("Bag traverser", "[to_struct]") {
	using at = noarr::array_t<'x', 2, noarr::array_t<'y', 3, noarr::scalar<int>>>;
	using bt = noarr::array_t<'y', 3, noarr::array_t<'z', 4, noarr::scalar<int>>>;

	auto a = at();
	auto b = bt();

	auto aw = make_bag(a, (char *)nullptr);
	auto bw = make_bag(b, (char *)nullptr);

	REQUIRE(std::is_empty_v<decltype(traverser(a, b))>);
	STATIC_REQUIRE(std::is_same_v<decltype(traverser(a, b)), decltype(traverser(aw, bw))>);

	REQUIRE(noarr_test::equal_data(traverser(a, b), traverser(aw, bw)));
}

TEST_CASE("Vector wrapper traverser", "[to_struct]") {
	using at = noarr::vector_t<'x', noarr::vector_t<'y', noarr::scalar<int>>>;
	using bt = noarr::vector_t<'y', noarr::vector_t<'z', noarr::scalar<int>>>;

	auto a = at() ^ noarr::set_length<'x', 'y'>(2, 3);
	auto b = bt() ^ noarr::set_length<'y', 'z'>(3, 4);

	auto aw = noarr::make_bag(a, (char *)nullptr);
	auto bw = noarr::make_bag(b, (char *)nullptr);
	STATIC_REQUIRE(std::is_same_v<decltype(aw), decltype(bag(a, (char *)nullptr))>);
	STATIC_REQUIRE(std::is_same_v<decltype(bw), decltype(bag(b, (char *)nullptr))>);

	REQUIRE(noarr_test::equal_data(traverser(a, b), traverser(aw, bw)));
}

TEST_CASE("Vector bag traverser", "[to_struct]") {
	using at = noarr::vector_t<'x', noarr::vector_t<'y', noarr::scalar<int>>>;
	using bt = noarr::vector_t<'y', noarr::vector_t<'z', noarr::scalar<int>>>;

	auto a = at() ^ noarr::set_length<'x', 'y'>(2, 3);
	auto b = bt() ^ noarr::set_length<'y', 'z'>(3, 4);

	auto aw = noarr::make_bag(a, (char *)nullptr);
	auto bw = noarr::make_bag(b, (char *)nullptr);

	REQUIRE(noarr_test::equal_data(traverser(a, b), traverser(aw, bw)));
}
