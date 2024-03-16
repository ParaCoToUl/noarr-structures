#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/planner_iter.hpp>

// TODO: add tests from traverser_iter_test.cpp

using namespace noarr;

TEST_CASE("Planner begin end", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	using iter_t = noarr::planner_iterator_t<'x', noarr::union_t<decltype(bag.get_ref())>, noarr::neutral_proto, noarr::planner_endings<>>;

	auto p = noarr::planner(bag);

	auto b = begin(p);
	auto e = end(p);

	STATIC_REQUIRE(std::is_same_v<decltype(b), iter_t>);
	STATIC_REQUIRE(std::is_same_v<decltype(e), iter_t>);

	REQUIRE(b.idx == 0);
	REQUIRE(e.idx == 20);
}

TEST_CASE("Planner trivial iteration", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	std::size_t x = 0, y = 0;
	for (auto col : p) {
		STATIC_REQUIRE(std::is_same_v<decltype(col), decltype(p ^ fix<'x'>(0))>);
		REQUIRE(noarr::get_index<'x'>(col.state()) == x++);
		for (auto cell : col) {
			STATIC_REQUIRE(std::is_same_v<decltype(cell), decltype(col ^ fix<'y'>(0))>);
			REQUIRE(noarr::get_index<'y'>(cell.state()) == y++);
		}
		y = 0;
	}
}

TEST_CASE("Planner iteration with hoist", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag) ^ hoist<'y'>();

	std::size_t x = 0, y = 0;
	for (auto col : p) {
		STATIC_REQUIRE(std::is_same_v<decltype(col), decltype(p ^ fix<'y'>(0))>);
		REQUIRE(noarr::get_index<'y'>(col.state()) == y++);
		for (auto cell : col) {
			STATIC_REQUIRE(std::is_same_v<decltype(cell), decltype(col ^ fix<'x'>(0))>);
			REQUIRE(noarr::get_index<'x'>(cell.state()) == x++);
		}
		x = 0;
	}
}
