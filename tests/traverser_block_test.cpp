#include "noarr/structures/extra/shortcuts.hpp"
#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/traverser_iter.hpp>

using namespace noarr;

TEST_CASE("Traverser block_static", "[traverser block]") {
	const auto s = scalar<int>() ^ vector<'x'>(22) ^ into_blocks_static<'x', 'b', 'X', 'x'>(10);
	const auto t = noarr::traverser(s);

	std::size_t b = 0;
	std::size_t X = 0;
	std::size_t x = 0;

	const auto test_lambda = [&b, &X, &x](auto state) {
		const auto [_b, _y, _x] = get_indices<'b', 'X', 'x'>(state);
		REQUIRE(_b == b);
		REQUIRE(_y == X);
		REQUIRE(_x == x);

		x++;
		if (x == 10) {
			x = 0;
			X++;
		}
		if (X == 2) {
			X = 0;
			b++;
		}
	};

	const auto test_lambda2 = [&b, &X, &x](auto state) {
		const auto [_b, _y, _x] = get_indices<'b', 'X', 'x'>(state);
		REQUIRE(_b == b);
		REQUIRE(_y == X);
		REQUIRE(_x == x);

		X++;
		if (b != 0 || X == 2) {
			X = 0;
			x++;
		}
		if (x == 10) {
			x = 0;
			b++;
		}
	};

	t | test_lambda;

	REQUIRE(b == 1);
	REQUIRE(X == 0);
	REQUIRE(x == 2);

	b = 0;
	X = 0;
	x = 0;

	t | for_each(test_lambda);

	REQUIRE(b == 1);
	REQUIRE(X == 0);
	REQUIRE(x == 2);

	b = 0;
	X = 0;
	x = 0;

	t | for_sections<'b', 'X', 'x'>(test_lambda);

	REQUIRE(b == 1);
	REQUIRE(X == 0);
	REQUIRE(x == 2);

	b = 0;
	X = 0;
	x = 0;

	t | for_sections<'b', 'x', 'X'>(test_lambda);

	REQUIRE(b == 1);
	REQUIRE(X == 0);
	REQUIRE(x == 2);

	b = 0;
	X = 0;
	x = 0;

	t | for_dims<'b', 'X', 'x'>(test_lambda);

	REQUIRE(b == 1);
	REQUIRE(X == 0);
	REQUIRE(x == 2);

	b = 0;
	X = 0;
	x = 0;

	t | for_dims<'b', 'x', 'X'>(test_lambda2);

	REQUIRE(b == 1);
	REQUIRE(X == 0);
	REQUIRE(x == 2);
}
