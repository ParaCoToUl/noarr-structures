#include <noarr_test/macros.hpp>

#include <noarr/structures.hpp>
#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/structs/bcast.hpp>

TEST_CASE("Broadcast offset and size", "[bcast]") {
	auto sa = noarr::scalar<float>() ^ noarr::sized_vector<'x'>(100);
	auto sb = sa ^ noarr::bcast<'y'>(150);
	REQUIRE((sa | noarr::offset<'x'>(10)) == (sb | noarr::offset<'x', 'y'>(10, 15)));
	REQUIRE((sa | noarr::get_size()) == (sb | noarr::get_size()));
	REQUIRE((sb | noarr::get_length<'x'>()) == 100);
	REQUIRE((sb | noarr::get_length<'y'>()) == 150);
}

TEST_CASE("Broadcast traverser", "[bcast traverser]") {
	auto sa = noarr::scalar<float>() ^ noarr::sized_vector<'x'>(10);
	auto sb = sa ^ noarr::bcast<'y'>(15);

	std::size_t x = 0, y = 0, total = 0;
	traverser(sb).for_each([&](auto state){
		REQUIRE(noarr::get_index<'x'>(state) == x);
		REQUIRE(noarr::get_index<'y'>(state) == y);
		total++;
		if(++x == 10)
			x = 0, ++y;
	});
	REQUIRE(x == 0);
	REQUIRE(y == 15);
	REQUIRE(total == 150);
}

TEST_CASE("Broadcast traverser order", "[bcast traverser]") {
	auto sa = noarr::scalar<float>() ^ noarr::sized_vector<'x'>(10);

	std::size_t x = 0, y = 0, total = 0;
	traverser(sa).order(noarr::bcast<'y'>(15)).for_each([&](auto state){
		REQUIRE(noarr::get_index<'x'>(state) == x);
		total++;
		if(++x == 10)
			x = 0, ++y;
	});
	REQUIRE(x == 0);
	REQUIRE(y == 15);
	REQUIRE(total == 150);
}
