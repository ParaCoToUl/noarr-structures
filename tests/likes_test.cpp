#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include <noarr/structures_extended.hpp>

TEST_CASE("Likes", "[shortcut]") {
	auto scalar = noarr::scalar<float>();

	auto sa = scalar ^ noarr::sized_vector<'x'>(100);
	auto sa_ = scalar ^ noarr::vector<'x'>();

	auto sb = scalar ^ noarr::sized_vector<'x'>(100) ^ noarr::sized_vector<'y'>(200);
	auto sb_ = scalar ^ noarr::vector<'x'>() ^ noarr::vector<'y'>();

	// reconstruct sa from scalar:
	REQUIRE(std::is_same_v<decltype(sa), decltype(scalar ^ noarr::vector_like<'x'>(sa))>);

	// reconstruct sa from scalar using dimensions from sb:
	REQUIRE(std::is_same_v<decltype(sa), decltype(scalar ^ noarr::vector_like<'x'>(sb))>);

	// reconstruct sa from sa_:
	REQUIRE(std::is_same_v<decltype(sa), decltype(sa_ ^ noarr::length_like<'x'>(sa))>);

	// reconstruct sa from sa_ using dimensions from sb:
	REQUIRE(std::is_same_v<decltype(sa), decltype(sa_ ^ noarr::length_like<'x'>(sb))>);

	// reconstruct sb from scalar:
	REQUIRE(std::is_same_v<decltype(sb), decltype(scalar ^ noarr::vectors_like<'x', 'y'>(sb))>);

	// reconstruct sb from sb_:
	auto re_sb = sb_ ^ noarr::lengths_like<'x', 'y'>(sb);
	REQUIRE(std::is_same_v<decltype(sb)::signature, decltype(re_sb)::signature>);
	REQUIRE((sb | noarr::get_size()) == (re_sb | noarr::get_size()));
	REQUIRE((sb | noarr::offset<'x', 'y'>(0, 1)) == (re_sb | noarr::offset<'x', 'y'>(0, 1)));
	REQUIRE((sb | noarr::offset<'x', 'y'>(1, 0)) == (re_sb | noarr::offset<'x', 'y'>(1, 0)));
}

TEST_CASE("Likes with state", "[shortcut]") {
	auto scalar = noarr::scalar<float>();

	auto sa = scalar ^ noarr::sized_vector<'x'>(100);
	auto sa_ = scalar ^ noarr::vector<'x'>();

	auto sb = scalar ^ noarr::sized_vector<'x'>(100) ^ noarr::sized_vector<'y'>(200);
	auto sb_ = scalar ^ noarr::vector<'x'>() ^ noarr::vector<'y'>();

	auto state = noarr::make_state<noarr::length_in<'x'>, noarr::length_in<'y'>>(100, 200);
	auto vector = noarr::scalar<void>() ^ noarr::vectors<'x', 'y'>();

	// reconstruct sa from scalar:
	REQUIRE(std::is_same_v<decltype(sa), decltype(scalar ^ noarr::vector_like<'x'>(vector, state))>);

	// reconstruct sa from scalar using sb_:
	REQUIRE(std::is_same_v<decltype(sa), decltype(scalar ^ noarr::vector_like<'x'>(sb_, state))>);

	// reconstruct sa from sa_:
	REQUIRE(std::is_same_v<decltype(sa), decltype(sa_ ^ noarr::length_like<'x'>(vector, state))>);

	// reconstruct sa from sa_ using dimensions from sb_:
	REQUIRE(std::is_same_v<decltype(sa), decltype(sa_ ^ noarr::length_like<'x'>(sb_, state))>);

	// reconstruct sb from scalar:
	REQUIRE(std::is_same_v<decltype(sb), decltype(scalar ^ noarr::vectors_like<'x', 'y'>(vector, state))>);

	// reconstruct sb from sb_:
	auto re_sb = sb_ ^ noarr::lengths_like<'x', 'y'>(vector, state);
	REQUIRE(std::is_same_v<decltype(sb)::signature, decltype(re_sb)::signature>);
	REQUIRE((sb | noarr::get_size()) == (re_sb | noarr::get_size()));
	REQUIRE((sb | noarr::offset<'x', 'y'>(0, 1)) == (re_sb | noarr::offset<'x', 'y'>(0, 1)));
	REQUIRE((sb | noarr::offset<'x', 'y'>(1, 0)) == (re_sb | noarr::offset<'x', 'y'>(1, 0)));
}
