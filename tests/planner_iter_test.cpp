#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/extra/planner.hpp>
#include <noarr/structures/interop/planner_iter.hpp>

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

TEST_CASE("Planner iter arithmetics", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	auto b = begin(p);
	auto e = end(p);

	REQUIRE(b + 0 == b);
	REQUIRE(b - 0 == b);

	REQUIRE(e + 0 == e);
	REQUIRE(e - 0 == e);

	REQUIRE(e - b == 20);
	REQUIRE(b - e == -20);

	REQUIRE(b + 20 == e);
	REQUIRE(e - 20 == b);

	auto lb = b++;
	REQUIRE(b.idx == 1);
	REQUIRE(lb.idx == 0);
	REQUIRE(lb + 1 == b);

	auto rb = ++b;
	REQUIRE(b.idx == 2);
	REQUIRE(rb.idx == 2);
	REQUIRE(rb == b);

	auto le = e--;
	REQUIRE(e.idx == 19);
	REQUIRE(le.idx == 20);
	REQUIRE(le - 1 == e);

	auto re = --e;
	REQUIRE(e.idx == 18);
	REQUIRE(re.idx == 18);
	REQUIRE(re == e);

	REQUIRE(lb <= rb);
	REQUIRE(rb >= lb);
	REQUIRE(!(lb > rb));
	REQUIRE(!(rb < lb));

	REQUIRE(b <= b);
	REQUIRE(b >= b);
	REQUIRE(!(b < b));
	REQUIRE(!(b > b));

	REQUIRE(lb < rb);
	REQUIRE(rb > lb);
	REQUIRE(!(lb >= rb));
	REQUIRE(!(rb <= lb));

	REQUIRE(lb != rb);
	REQUIRE(rb != lb);
	REQUIRE(!(lb == rb));
	REQUIRE(!(rb == lb));

	rb -= 2;
	REQUIRE(rb == lb);
	REQUIRE(!(rb != lb));

	re += 2;
	REQUIRE(re == le);
	REQUIRE(!(re != le));

	rb += 10;
	re -= 10;
	REQUIRE(rb == re);
	REQUIRE((lb < b && b < rb && re < e && e < le));
}

TEST_CASE("Planner iter type traits", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	using iter_t = noarr::planner_iterator_t<'x', noarr::union_t<decltype(bag.get_ref())>, noarr::neutral_proto, noarr::planner_endings<>>;

	// iterator should only contain the structure (empty), pointer and index
	STATIC_REQUIRE(sizeof(iter_t) == sizeof(std::size_t) + sizeof(const void*));

	// like noarr_test::is_simple, but in addition require assignability
	// and do *not* require the *absence* of default ctor (since its *presence* is required by std::random_access_iterator)
	STATIC_REQUIRE(std::is_trivially_copy_constructible_v<iter_t>);
	STATIC_REQUIRE(std::is_trivially_move_constructible_v<iter_t>);
	STATIC_REQUIRE(std::is_trivially_copy_assignable_v<iter_t>);
	STATIC_REQUIRE(std::is_trivially_move_assignable_v<iter_t>);
	STATIC_REQUIRE(std::is_trivially_destructible_v<iter_t>);

	STATIC_REQUIRE(std::random_access_iterator<iter_t>);
}

TEST_CASE("Planner iter deref", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	std::size_t x = 7;
	auto p_x = begin(p)[x];

	std::size_t y = 0;
	p_x.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		REQUIRE(state.template get<index_in<'y'>>() == y);
		y++;
	})();
	REQUIRE(y == 30);
}

TEST_CASE("Planner iter deref deref", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	std::size_t x = 7;
	std::size_t y = 9;
	auto p_x_y = begin(begin(p)[x])[y];

	std::size_t z = 0;
	p_x_y.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		REQUIRE(state.template get<index_in<'y'>>() == y);
		z++;
	})();
	REQUIRE(z == 1);
	auto only_state = p_x_y.state();
	REQUIRE(only_state.template get<index_in<'x'>>() == x);
	REQUIRE(only_state.template get<index_in<'y'>>() == y);
}

TEST_CASE("Planner iter for", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	std::size_t x = 0;
	for(auto p_x : p) {
		std::size_t y = 0;
		p_x.for_each([&](auto state){
			REQUIRE(state.template get<index_in<'x'>>() == x);
			REQUIRE(state.template get<index_in<'y'>>() == y);
			y++;
		})();
		REQUIRE(y == 30);
		x++;
	}
	REQUIRE(x == 20);
}

TEST_CASE("Planner iter for for", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	std::size_t x = 0;
	for(auto p_x : p) {
		std::size_t y = 0;
		for(auto p_x_y : p_x) {
			std::size_t z = 0;
			p_x_y.for_each([&](auto state){
				REQUIRE(state.template get<index_in<'x'>>() == x);
				REQUIRE(state.template get<index_in<'y'>>() == y);
				z++;
			})();
			REQUIRE(z == 1);
			auto only_state = p_x_y.state();
			REQUIRE(only_state.template get<index_in<'x'>>() == x);
			REQUIRE(only_state.template get<index_in<'y'>>() == y);
			y++;
		}
		REQUIRE(y == 30);
		x++;
	}
	REQUIRE(x == 20);
}

TEST_CASE("Planner trivial iteration", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	std::size_t x = 0, y = 0;
	for (auto col : p) {
		STATIC_REQUIRE(std::is_same_v<decltype(col), decltype(p ^ fix<'x'>(0))>);
		REQUIRE(noarr::get_index<'x'>(col) == x++);
		for (auto cell : col) {
			STATIC_REQUIRE(std::is_same_v<decltype(cell), decltype(col ^ fix<'y'>(0))>);
			REQUIRE(noarr::get_index<'y'>(cell) == y++);
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

// TESTS FOR RANGE

TEST_CASE("Planner range methods", "[planner iter]") {
	auto matrix_bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto array_bag = make_bag(scalar<int>() ^ array<'x', 1>(), (const void*)nullptr);
	auto empty_bag = make_bag(scalar<int>() ^ array<'x', 0>(), (const void*)nullptr);

	{
		auto p = noarr::planner(matrix_bag);
		auto r = range(p);

		REQUIRE(r.size() == 20);
		REQUIRE(r.begin() == begin(p));
		REQUIRE(r.end() == end(p));
		REQUIRE(r.empty() == false);
		REQUIRE(r.is_divisible() == true);
	}

	{
		auto p = noarr::planner(array_bag);
		auto r = range(p);

		REQUIRE(r.size() == 1);
		REQUIRE(r.begin() == begin(p));
		REQUIRE(r.end() == end(p));
		REQUIRE(r.empty() == false);
		REQUIRE(r.is_divisible() == false);
	}

	{
		auto p = noarr::planner(empty_bag);
		auto r = range(p);

		REQUIRE(r.size() == 0);
		REQUIRE(r.begin() == begin(p));
		REQUIRE(r.end() == end(p));
		REQUIRE(r.begin() == r.end());
		REQUIRE(r.empty() == true);
		REQUIRE(r.is_divisible() == false);
	}
}

TEST_CASE("Planner range deref", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	std::size_t x = 7;
	auto p_x = range<'x'>(p)[x];

	std::size_t y = 0;
	p_x.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		REQUIRE(state.template get<index_in<'y'>>() == y);
		y++;
	})();
	REQUIRE(y == 30);
}

TEST_CASE("Planner range for", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	std::size_t x = 0;
	for(auto p_x : range<'x'>(p)) {
		std::size_t y = 0;
		p_x.for_each([&](auto state){
			REQUIRE(state.template get<index_in<'x'>>() == x);
			REQUIRE(state.template get<index_in<'y'>>() == y);
			y++;
		})();
		REQUIRE(y == 30);
		x++;
	}
	REQUIRE(x == 20);
}

TEST_CASE("Planner subrange deref", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	auto r = range<'x'>(p);
	r.begin_idx = 3;
	r.end_idx = 17;

	std::size_t x = 7;
	auto p_x = r[x-3];

	std::size_t y = 0;
	p_x.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		REQUIRE(state.template get<index_in<'y'>>() == y);
		y++;
	})();
	REQUIRE(y == 30);
}

TEST_CASE("Planner subrange for", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'y', 30>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	auto r = range<'x'>(p);
	r.begin_idx = 3;
	r.end_idx = 17;

	std::size_t x = 3;
	for(auto p_x : r) {
		std::size_t y = 0;
		p_x.for_each([&](auto state){
			REQUIRE(state.template get<index_in<'x'>>() == x);
			REQUIRE(state.template get<index_in<'y'>>() == y);
			y++;
		})();
		REQUIRE(y == 30);
		x++;
	}
	REQUIRE(x == 17);
}

TEST_CASE("Planner subrange for_each", "[planner iter]") {
	auto bag = make_bag(scalar<int>() ^ array<'x', 20>(), (const void*)nullptr);
	auto p = noarr::planner(bag);

	auto r = range<'x'>(p);
	r.begin_idx = 3;
	r.end_idx = 17;

	std::size_t x = 3;
	r.for_each([&](auto state){
		REQUIRE(state.template get<index_in<'x'>>() == x);
		x++;
	})();
	REQUIRE(x == 17);
}
