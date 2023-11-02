#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/structs/slice.hpp>

TEST_CASE("Step", "[step]") {
	auto m = noarr::scalar<float>() ^ noarr::array<'x', 10>();
	auto m0 = m ^ noarr::step<'x'>(0, 4);
	auto m1 = m ^ noarr::step<'x'>(1, 4);
	auto m2 = m ^ noarr::step<'x'>(2, 4);
	auto m3 = m ^ noarr::step<'x'>(3, 4);

	REQUIRE((m0 | noarr::get_length<'x'>()) == 3);
	REQUIRE((m1 | noarr::get_length<'x'>()) == 3);
	REQUIRE((m2 | noarr::get_length<'x'>()) == 2);
	REQUIRE((m3 | noarr::get_length<'x'>()) == 2);

	REQUIRE((m0 | noarr::offset<'x'>(0)) == 0 * sizeof(float));
	REQUIRE((m1 | noarr::offset<'x'>(0)) == 1 * sizeof(float));
	REQUIRE((m2 | noarr::offset<'x'>(0)) == 2 * sizeof(float));
	REQUIRE((m3 | noarr::offset<'x'>(0)) == 3 * sizeof(float));
	REQUIRE((m0 | noarr::offset<'x'>(1)) == 4 * sizeof(float));
	REQUIRE((m1 | noarr::offset<'x'>(1)) == 5 * sizeof(float));
	REQUIRE((m2 | noarr::offset<'x'>(1)) == 6 * sizeof(float));
	REQUIRE((m3 | noarr::offset<'x'>(1)) == 7 * sizeof(float));
	REQUIRE((m0 | noarr::offset<'x'>(2)) == 8 * sizeof(float));
	REQUIRE((m1 | noarr::offset<'x'>(2)) == 9 * sizeof(float));
}

TEST_CASE("Auto step", "[step]") {
	auto m = noarr::scalar<float>() ^ noarr::array<'x', 10>();
	auto m0 = m ^ noarr::step(0, 4);
	auto m1 = m ^ noarr::step(1, 4);
	auto m2 = m ^ noarr::step(2, 4);
	auto m3 = m ^ noarr::step(3, 4);

	REQUIRE((m0 | noarr::get_length<'x'>()) == 3);
	REQUIRE((m1 | noarr::get_length<'x'>()) == 3);
	REQUIRE((m2 | noarr::get_length<'x'>()) == 2);
	REQUIRE((m3 | noarr::get_length<'x'>()) == 2);

	REQUIRE((m0 | noarr::offset<'x'>(0)) == 0 * sizeof(float));
	REQUIRE((m1 | noarr::offset<'x'>(0)) == 1 * sizeof(float));
	REQUIRE((m2 | noarr::offset<'x'>(0)) == 2 * sizeof(float));
	REQUIRE((m3 | noarr::offset<'x'>(0)) == 3 * sizeof(float));
	REQUIRE((m0 | noarr::offset<'x'>(1)) == 4 * sizeof(float));
	REQUIRE((m1 | noarr::offset<'x'>(1)) == 5 * sizeof(float));
	REQUIRE((m2 | noarr::offset<'x'>(1)) == 6 * sizeof(float));
	REQUIRE((m3 | noarr::offset<'x'>(1)) == 7 * sizeof(float));
	REQUIRE((m0 | noarr::offset<'x'>(2)) == 8 * sizeof(float));
	REQUIRE((m1 | noarr::offset<'x'>(2)) == 9 * sizeof(float));
}
