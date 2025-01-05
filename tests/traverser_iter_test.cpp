#include <noarr_test/macros.hpp>

#include <concepts>
#include <iterator>
#include <ranges>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>

using namespace noarr;

TEST_CASE("Traverser begin end", "[traverser iter]") {
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;
	using iter_t = noarr::traverser_iterator_t<'x', noarr::union_t<s>, noarr::neutral_proto>;

	auto t = noarr::traverser(s());

	auto b = t.begin();
	auto e = t.end();

	STATIC_REQUIRE(std::is_same_v<decltype(b), iter_t>);
	STATIC_REQUIRE(std::is_same_v<decltype(e), iter_t>);

	REQUIRE(b.idx == 0);
	REQUIRE(e.idx == 20);
}

TEST_CASE("Traverser iter arithmetics", "[traverser iter]") {
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	auto b = t.begin();
	auto e = t.end();

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

TEST_CASE("Traverser iter type traits", "[traverser iter]") {
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;
	using iter_t = noarr::traverser_iterator_t<'x', noarr::union_t<s>, noarr::neutral_proto>;

	// iterator should only contain the structure (empty) and index
	STATIC_REQUIRE(sizeof(iter_t) == sizeof(std::size_t));

	// like noarr_test::is_simple, but in addition require assignability
	// and do *not* require the *absence* of default ctor (since its *presence* is required by std::random_access_iterator)
	STATIC_REQUIRE(std::is_trivially_copy_constructible_v<iter_t>);
	STATIC_REQUIRE(std::is_trivially_move_constructible_v<iter_t>);
	STATIC_REQUIRE(std::is_trivially_copy_assignable_v<iter_t>);
	STATIC_REQUIRE(std::is_trivially_move_assignable_v<iter_t>);
	STATIC_REQUIRE(std::is_trivially_destructible_v<iter_t>);

	STATIC_REQUIRE(std::random_access_iterator<iter_t>);
}

TEST_CASE("Traverser iter deref", "[traverser iter]") {
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

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
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

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
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

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
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

	auto t = noarr::traverser(s());

	// check if the iterators are random access
	static_assert(std::random_access_iterator<decltype(t.begin())>);
	static_assert(std::random_access_iterator<decltype(t.end())>);

	// check if the traverser models the random access range concept
	static_assert(std::ranges::range<decltype(t)>);

	if constexpr (std::ranges::range<decltype(t)>) {
		static_assert(std::ranges::random_access_range<decltype(t)>);
	}

	std::size_t x = 0;
	for(auto t_x : t) {
		static_assert(std::random_access_iterator<decltype(t_x.begin())>);
		static_assert(std::random_access_iterator<decltype(t_x.end())>);

		static_assert(std::ranges::range<decltype(t_x)>);

		if constexpr (std::ranges::range<decltype(t_x)>) {
			static_assert(std::ranges::random_access_range<decltype(t_x)>);
		}

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

// TESTS FOR RANGE

TEST_CASE("Traverser range methods", "[traverser iter]") {
	using matrix = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;
	using array = noarr::array_t<'x', 1, noarr::scalar<int>>;
	using empty = noarr::array_t<'x', 0, noarr::scalar<int>>;

	{
		auto t = noarr::traverser(matrix());
		auto r = t.range();

		REQUIRE(r.size() == 20);
		REQUIRE(r.begin() == t.begin());
		REQUIRE(r.end() == t.end());
		REQUIRE(r.empty() == false);
		REQUIRE(r.is_divisible() == true);
	}

	{
		auto t = noarr::traverser(array());
		auto r = t.range();

		REQUIRE(r.size() == 1);
		REQUIRE(r.begin() == t.begin());
		REQUIRE(r.end() == t.end());
		REQUIRE(r.empty() == false);
		REQUIRE(r.is_divisible() == false);
	}

	{
		auto t = noarr::traverser(empty());
		auto r = t.range();

		REQUIRE(r.size() == 0);
		REQUIRE(r.begin() == t.begin());
		REQUIRE(r.end() == t.end());
		REQUIRE(r.begin() == r.end());
		REQUIRE(r.empty() == true);
		REQUIRE(r.is_divisible() == false);
	}
}

TEST_CASE("Traverser range deref", "[traverser iter]") {
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

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
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

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
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

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
	using s = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

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
	using s = noarr::array_t<'x', 20, noarr::scalar<int>>;

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
