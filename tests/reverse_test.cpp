#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/structs/slice.hpp>
#include <noarr/structures/extra/traverser.hpp>

using noarr::lit;

static auto lenstate = noarr::make_state<noarr::length_in<'x'>>(42);

static auto mkstate(std::size_t x) {
	return noarr::make_state<noarr::length_in<'x'>, noarr::index_in<'x'>>(42, x);
}

TEST_CASE("Reverse sized vector", "[reverse]") {
	auto s = noarr::scalar<float>() ^ noarr::sized_vector<'x'>(42);
	auto r = s ^ noarr::reverse<'x'>();

	REQUIRE((std::is_same_v<decltype(r)::signature, decltype(s)::signature>));
	REQUIRE((s | noarr::offset<'x'>(0)) == (r | noarr::offset<'x'>(41)));
	REQUIRE((s | noarr::offset<'x'>(10)) == (r | noarr::offset<'x'>(41-10)));
	REQUIRE((s | noarr::get_length<'x'>()) == (r | noarr::get_length<'x'>()));
}

TEST_CASE("Reverse unsized vector", "[reverse]") {
	auto s = noarr::scalar<float>() ^ noarr::vector<'x'>();
	auto r = s ^ noarr::reverse<'x'>();

	REQUIRE((std::is_same_v<decltype(r)::signature, decltype(s)::signature>));
	REQUIRE((s | noarr::offset(mkstate(0))) == (r | noarr::offset(mkstate(41))));
	REQUIRE((s | noarr::offset(mkstate(10))) == (r | noarr::offset(mkstate(41-10))));
	REQUIRE((s | noarr::get_length<'x'>(lenstate)) == (r | noarr::get_length<'x'>(lenstate)));
}

TEST_CASE("Reverse array", "[reverse]") {
	auto s = noarr::scalar<float>() ^ noarr::array<'x', 42>();
	auto r = s ^ noarr::reverse<'x'>();

	REQUIRE((std::is_same_v<decltype(r)::signature, decltype(s)::signature>));
	REQUIRE((s | noarr::offset<'x'>(0)) == (r | noarr::offset<'x'>(41)));
	REQUIRE((s | noarr::offset<'x'>(10)) == (r | noarr::offset<'x'>(41-10)));
	REQUIRE((s | noarr::offset<'x'>(lit<0>)).value == (r | noarr::offset<'x'>(lit<41>)).value);
	REQUIRE((s | noarr::offset<'x'>(lit<10>)).value == (r | noarr::offset<'x'>(lit<41-10>)).value);
	REQUIRE((s | noarr::get_length<'x'>()).value == (r | noarr::get_length<'x'>()).value);
}

TEST_CASE("Reverse tuple", "[reverse]") {
	struct item0 { float field0; };
	struct item1 { float field1; };
	struct item2 { float field2; };
	auto s = noarr::pack(noarr::scalar<item0>(), noarr::scalar<item1>(), noarr::scalar<item2>()) ^ noarr::tuple<'x'>();
	auto r = s ^ noarr::reverse<'x'>();
	auto q = noarr::pack(noarr::scalar<item2>(), noarr::scalar<item1>(), noarr::scalar<item0>()) ^ noarr::tuple<'x'>();

	REQUIRE(!(std::is_same_v<decltype(r)::signature, decltype(s)::signature>));
	REQUIRE((std::is_same_v<decltype(r)::signature, decltype(q)::signature>));
	REQUIRE((s | noarr::offset<'x'>(lit<0>)).value == (r | noarr::offset<'x'>(lit<2>)).value);
	REQUIRE((s | noarr::offset<'x'>(lit<1>)).value == (r | noarr::offset<'x'>(lit<1>)).value);
	REQUIRE((s | noarr::offset<'x'>(lit<2>)).value == (r | noarr::offset<'x'>(lit<0>)).value);
	REQUIRE((s | noarr::get_length<'x'>()).value == (r | noarr::get_length<'x'>()).value);
}

TEST_CASE("Reverse sized vector traverser", "[reverse traverser]") {
	auto s = noarr::scalar<float>() ^ noarr::sized_vector<'x'>(42);
	auto r = s ^ noarr::reverse<'x'>();

	std::size_t x = 0;
	noarr::traverser(r).for_each([&](auto state) {
		REQUIRE(noarr::get_index<'x'>(state) == x);
		REQUIRE((r | noarr::offset(state)) == (s | noarr::offset<'x'>(41-x)));
		x++;
	});

	REQUIRE(x == 42);

	noarr::traverser(s).order(noarr::reverse<'x'>()).for_each([&](auto state) {
		x--;
		REQUIRE(noarr::get_index<'x'>(state) == x);
	});

	REQUIRE(x == 0);
}

TEST_CASE("Reverse unsized vector traverser", "[reverse traverser]") {
	auto s = noarr::scalar<float>() ^ noarr::vector<'x'>();
	auto r = s ^ noarr::reverse<'x'>();

	std::size_t x = 0;
	noarr::traverser(r).order(noarr::set_length<'x'>(42)).for_each([&](auto state) {
		REQUIRE(noarr::get_index<'x'>(state) == x);
		REQUIRE((r | noarr::offset(state)) == (s | noarr::offset(mkstate(41-x))));
		x++;
	});

	REQUIRE(x == 42);

	noarr::traverser(s).order(noarr::reverse<'x'>() ^ noarr::set_length<'x'>(42)).for_each([&](auto state) {
		x--;
		REQUIRE(noarr::get_index<'x'>(state) == x);
	});

	REQUIRE(x == 0);
}

TEST_CASE("Reverse array traverser", "[reverse traverser]") {
	auto s = noarr::scalar<float>() ^ noarr::array<'x', 42>();
	auto r = s ^ noarr::reverse<'x'>();

	std::size_t x = 0;
	noarr::traverser(r).for_each([&](auto state) {
		REQUIRE(noarr::get_index<'x'>(state) == x);
		REQUIRE((r | noarr::offset(state)) == (s | noarr::offset<'x'>(41-x)));
		x++;
	});

	REQUIRE(x == 42);

	noarr::traverser(s).order(noarr::reverse<'x'>()).for_each([&](auto state) {
		x--;
		REQUIRE(noarr::get_index<'x'>(state) == x);
	});

	REQUIRE(x == 0);
}

TEST_CASE("Reverse tuple traverser", "[reverse traverser]") {
	struct item0 { float field0; };
	struct item1 { float field1; };
	struct item2 { float field2; };
	auto s = noarr::pack(noarr::scalar<item0>(), noarr::scalar<item1>(), noarr::scalar<item2>()) ^ noarr::tuple<'x'>();
	auto r = s ^ noarr::reverse<'x'>();

	REQUIRE((s | noarr::get_length<'x'>()) == 3);
	REQUIRE((r | noarr::get_length<'x'>()) == 3);

	std::size_t x = 0;
	noarr::traverser(r).for_each([&](auto state) {
		constexpr auto state_x = decltype(noarr::get_index<'x'>(state))::value;
		REQUIRE(state_x == x);
		REQUIRE((r | noarr::offset(state)) == (s | noarr::offset<'x'>(lit<2-state_x>)));
		x++;
	});

	REQUIRE(x == 3);

	noarr::traverser(s).order(noarr::reverse<'x'>()).for_each([&](auto state) {
		x--;
		REQUIRE(noarr::get_index<'x'>(state).value == x);
	});

	REQUIRE(x == 0);
}
