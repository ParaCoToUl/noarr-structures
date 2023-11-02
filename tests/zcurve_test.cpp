#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/structs/zcurve.hpp>

TEST_CASE("Z curve", "[zcurve]") {
	auto a = noarr::array_t<'y', 4, noarr::array_t<'x', 4, noarr::scalar<int>>>();
	auto z = a ^ noarr::merge_zcurve<'y', 'x', 'z'>::maxlen_alignment<4, 2>();

	REQUIRE((z | noarr::offset<'z'>( 0)) == (a | noarr::offset<'x', 'y'>(0, 0)));
	REQUIRE((z | noarr::offset<'z'>( 1)) == (a | noarr::offset<'x', 'y'>(1, 0)));
	REQUIRE((z | noarr::offset<'z'>( 2)) == (a | noarr::offset<'x', 'y'>(0, 1)));
	REQUIRE((z | noarr::offset<'z'>( 3)) == (a | noarr::offset<'x', 'y'>(1, 1)));

	REQUIRE((z | noarr::offset<'z'>( 4)) == (a | noarr::offset<'x', 'y'>(2, 0)));
	REQUIRE((z | noarr::offset<'z'>( 5)) == (a | noarr::offset<'x', 'y'>(3, 0)));
	REQUIRE((z | noarr::offset<'z'>( 6)) == (a | noarr::offset<'x', 'y'>(2, 1)));
	REQUIRE((z | noarr::offset<'z'>( 7)) == (a | noarr::offset<'x', 'y'>(3, 1)));

	REQUIRE((z | noarr::offset<'z'>( 8)) == (a | noarr::offset<'x', 'y'>(0, 2)));
	REQUIRE((z | noarr::offset<'z'>( 9)) == (a | noarr::offset<'x', 'y'>(1, 2)));
	REQUIRE((z | noarr::offset<'z'>(10)) == (a | noarr::offset<'x', 'y'>(0, 3)));
	REQUIRE((z | noarr::offset<'z'>(11)) == (a | noarr::offset<'x', 'y'>(1, 3)));

	REQUIRE((z | noarr::offset<'z'>(12)) == (a | noarr::offset<'x', 'y'>(2, 2)));
	REQUIRE((z | noarr::offset<'z'>(13)) == (a | noarr::offset<'x', 'y'>(3, 2)));
	REQUIRE((z | noarr::offset<'z'>(14)) == (a | noarr::offset<'x', 'y'>(2, 3)));
	REQUIRE((z | noarr::offset<'z'>(15)) == (a | noarr::offset<'x', 'y'>(3, 3)));
}

TEST_CASE("Z curve misaligned", "[zcurve]") {
	auto a = noarr::array_t<'y', 6, noarr::array_t<'x', 6, noarr::scalar<int>>>();
	auto z = a ^ noarr::merge_zcurve<'y', 'x', 'z'>::maxlen_alignment<8, 2>();

	REQUIRE((z | noarr::offset<'z'>( 0)) == (a | noarr::offset<'x', 'y'>(0, 0)));
	REQUIRE((z | noarr::offset<'z'>( 1)) == (a | noarr::offset<'x', 'y'>(1, 0)));
	REQUIRE((z | noarr::offset<'z'>( 2)) == (a | noarr::offset<'x', 'y'>(0, 1)));
	REQUIRE((z | noarr::offset<'z'>( 3)) == (a | noarr::offset<'x', 'y'>(1, 1)));

	REQUIRE((z | noarr::offset<'z'>( 4)) == (a | noarr::offset<'x', 'y'>(2, 0)));
	REQUIRE((z | noarr::offset<'z'>( 5)) == (a | noarr::offset<'x', 'y'>(3, 0)));
	REQUIRE((z | noarr::offset<'z'>( 6)) == (a | noarr::offset<'x', 'y'>(2, 1)));
	REQUIRE((z | noarr::offset<'z'>( 7)) == (a | noarr::offset<'x', 'y'>(3, 1)));

	REQUIRE((z | noarr::offset<'z'>( 8)) == (a | noarr::offset<'x', 'y'>(0, 2)));
	REQUIRE((z | noarr::offset<'z'>( 9)) == (a | noarr::offset<'x', 'y'>(1, 2)));
	REQUIRE((z | noarr::offset<'z'>(10)) == (a | noarr::offset<'x', 'y'>(0, 3)));
	REQUIRE((z | noarr::offset<'z'>(11)) == (a | noarr::offset<'x', 'y'>(1, 3)));

	REQUIRE((z | noarr::offset<'z'>(12)) == (a | noarr::offset<'x', 'y'>(2, 2)));
	REQUIRE((z | noarr::offset<'z'>(13)) == (a | noarr::offset<'x', 'y'>(3, 2)));
	REQUIRE((z | noarr::offset<'z'>(14)) == (a | noarr::offset<'x', 'y'>(2, 3)));
	REQUIRE((z | noarr::offset<'z'>(15)) == (a | noarr::offset<'x', 'y'>(3, 3)));

	REQUIRE((z | noarr::offset<'z'>(16)) == (a | noarr::offset<'x', 'y'>(4, 0)));
	REQUIRE((z | noarr::offset<'z'>(17)) == (a | noarr::offset<'x', 'y'>(5, 0)));
	REQUIRE((z | noarr::offset<'z'>(18)) == (a | noarr::offset<'x', 'y'>(4, 1)));
	REQUIRE((z | noarr::offset<'z'>(19)) == (a | noarr::offset<'x', 'y'>(5, 1)));

	REQUIRE((z | noarr::offset<'z'>(20)) == (a | noarr::offset<'x', 'y'>(4, 2)));
	REQUIRE((z | noarr::offset<'z'>(21)) == (a | noarr::offset<'x', 'y'>(5, 2)));
	REQUIRE((z | noarr::offset<'z'>(22)) == (a | noarr::offset<'x', 'y'>(4, 3)));
	REQUIRE((z | noarr::offset<'z'>(23)) == (a | noarr::offset<'x', 'y'>(5, 3)));

	REQUIRE((z | noarr::offset<'z'>(24)) == (a | noarr::offset<'x', 'y'>(0, 4)));
	REQUIRE((z | noarr::offset<'z'>(25)) == (a | noarr::offset<'x', 'y'>(1, 4)));
	REQUIRE((z | noarr::offset<'z'>(26)) == (a | noarr::offset<'x', 'y'>(0, 5)));
	REQUIRE((z | noarr::offset<'z'>(27)) == (a | noarr::offset<'x', 'y'>(1, 5)));

	REQUIRE((z | noarr::offset<'z'>(28)) == (a | noarr::offset<'x', 'y'>(2, 4)));
	REQUIRE((z | noarr::offset<'z'>(29)) == (a | noarr::offset<'x', 'y'>(3, 4)));
	REQUIRE((z | noarr::offset<'z'>(30)) == (a | noarr::offset<'x', 'y'>(2, 5)));
	REQUIRE((z | noarr::offset<'z'>(31)) == (a | noarr::offset<'x', 'y'>(3, 5)));

	REQUIRE((z | noarr::offset<'z'>(32)) == (a | noarr::offset<'x', 'y'>(4, 4)));
	REQUIRE((z | noarr::offset<'z'>(33)) == (a | noarr::offset<'x', 'y'>(5, 4)));
	REQUIRE((z | noarr::offset<'z'>(34)) == (a | noarr::offset<'x', 'y'>(4, 5)));
	REQUIRE((z | noarr::offset<'z'>(35)) == (a | noarr::offset<'x', 'y'>(5, 5)));
}
