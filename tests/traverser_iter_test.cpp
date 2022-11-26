#include <catch2/catch.hpp>

#include <array>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>

using namespace noarr;

TEST_CASE("Traverser begin end", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;
	using iter_t = noarr::traverser_iterator_t<'x', noarr::union_t<s>, noarr::neutral_proto>;

	auto t = noarr::traverser(s());

	auto b = t.begin();
	auto e = t.end();

	REQUIRE(std::is_same_v<decltype(b), iter_t>);
	REQUIRE(std::is_same_v<decltype(e), iter_t>);

	REQUIRE(b.idx == 0);
	REQUIRE(e.idx == 20);
}

TEST_CASE("Traverser iter type traits", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;
	using iter_t = noarr::traverser_iterator_t<'x', noarr::union_t<s>, noarr::neutral_proto>;

	// iterator should only contain the structure (empty) and index
	REQUIRE(sizeof(iter_t) == sizeof(std::size_t));

	// like noarr_test::is_simple, but in addition require assignability
	// and do *not* require the *absence* of default ctor (since its *presence* is required by std::random_access_iterator)
	REQUIRE(std::is_standard_layout_v<iter_t>);
	REQUIRE(std::is_trivially_copy_constructible_v<iter_t>);
	REQUIRE(std::is_trivially_move_constructible_v<iter_t>);
	REQUIRE(std::is_trivially_copy_assignable_v<iter_t>);
	REQUIRE(std::is_trivially_move_assignable_v<iter_t>);
	REQUIRE(std::is_trivially_destructible_v<iter_t>);

	// C++20:
	//REQUIRE(std::random_access_iterator<iter_t>);
}

TEST_CASE("Traverser iter deref", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	std::size_t x = 7;
	auto t_x = t.begin()[x];

	std::size_t y = 0;
	t_x.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		REQUIRE(state.template get<index_in<'y'>>() == y);
		y++;
	});
	REQUIRE(y == 30);
}

TEST_CASE("Traverser iter deref deref", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	std::size_t x = 7;
	std::size_t y = 9;
	auto t_x_y = t.begin()[x].begin()[y];

	std::size_t z = 0;
	t_x_y.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		REQUIRE(state.template get<index_in<'y'>>() == y);
		z++;
	});
	REQUIRE(z == 1);
	auto only_state = t_x_y.state();
	REQUIRE(only_state.template get<index_in<'x'>>() == x);
	REQUIRE(only_state.template get<index_in<'y'>>() == y);
}

TEST_CASE("Traverser iter for", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	std::size_t x = 0;
	for(auto t_x : t) {
		std::size_t y = 0;
		t_x.for_each([&](auto state){
			REQUIRE(state.template get<index_in<'x'>>() == x);
			REQUIRE(state.template get<index_in<'y'>>() == y);
			y++;
		});
		REQUIRE(y == 30);
		x++;
	}
	REQUIRE(x == 20);
}

TEST_CASE("Traverser iter for for", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	std::size_t x = 0;
	for(auto t_x : t) {
		std::size_t y = 0;
		for(auto t_x_y : t_x) {
			std::size_t z = 0;
			t_x_y.for_each([&](auto state){
				REQUIRE(state.template get<index_in<'x'>>() == x);
				REQUIRE(state.template get<index_in<'y'>>() == y);
				z++;
			});
			REQUIRE(z == 1);
			auto only_state = t_x_y.state();
			REQUIRE(only_state.template get<index_in<'x'>>() == x);
			REQUIRE(only_state.template get<index_in<'y'>>() == y);
			y++;
		}
		REQUIRE(y == 30);
		x++;
	}
	REQUIRE(x == 20);
}

TEST_CASE("Traverser range deref", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	std::size_t x = 7;
	auto t_x = t.range()[x];

	std::size_t y = 0;
	t_x.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		REQUIRE(state.template get<index_in<'y'>>() == y);
		y++;
	});
	REQUIRE(y == 30);
}

TEST_CASE("Traverser range for", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	std::size_t x = 0;
	for(auto t_x : t.range()) {
		std::size_t y = 0;
		t_x.for_each([&](auto state){
			REQUIRE(state.template get<index_in<'x'>>() == x);
			REQUIRE(state.template get<index_in<'y'>>() == y);
			y++;
		});
		REQUIRE(y == 30);
		x++;
	}
	REQUIRE(x == 20);
}

TEST_CASE("Traverser subrange deref", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	auto r = t.range();
	r.begin_idx = 3;
	r.end_idx = 17;

	std::size_t x = 7;
	auto t_x = r[x-3];

	std::size_t y = 0;
	t_x.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		REQUIRE(state.template get<index_in<'y'>>() == y);
		y++;
	});
	REQUIRE(y == 30);
}

TEST_CASE("Traverser subrange for", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	auto r = t.range();
	r.begin_idx = 3;
	r.end_idx = 17;

	std::size_t x = 3;
	for(auto t_x : r) {
		std::size_t y = 0;
		t_x.for_each([&](auto state){
			REQUIRE(state.template get<index_in<'x'>>() == x);
			REQUIRE(state.template get<index_in<'y'>>() == y);
			y++;
		});
		REQUIRE(y == 30);
		x++;
	}
	REQUIRE(x == 17);
}

TEST_CASE("Traverser subrange for_each", "[traverser iter]") {
	using s = noarr::array<'x', 20, noarr::scalar<int>>;

	auto t = noarr::traverser(s());

	auto r = t.range();
	r.begin_idx = 3;
	r.end_idx = 17;

	std::size_t x = 3;
	r.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		x++;
	});
	REQUIRE(x == 17);
}
