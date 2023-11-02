#include <noarr_test/macros.hpp>

#include <noarr/structures.hpp>
#include <noarr/structures/structs/blocks.hpp>
#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures/extra/traverser.hpp>

using noarr::idx;
using noarr::lit;

TEST_CASE("Blocks with static border", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'013>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_static<'x', 'c', 'b', 'a'>(16);

	REQUIRE(decltype(m)::signature::all_accept<'y'>);
	REQUIRE(decltype(m)::signature::all_accept<'c'>);
	REQUIRE(decltype(m)::signature::all_accept<'b'>);
	REQUIRE(decltype(m)::signature::all_accept<'a'>);

	REQUIRE((m | noarr::get_length<'y'>()) == 20'000);
	REQUIRE((m | noarr::get_length<'c'>()) == 2);
	REQUIRE((m | noarr::get_length<'b'>(idx<'c'>(lit<0>))) == 10'013 / 16);
	REQUIRE((m | noarr::get_length<'b'>(idx<'c'>(lit<1>))) ==           1);
	REQUIRE((m | noarr::get_length<'a'>(idx<'c'>(lit<0>))) ==          16);
	REQUIRE((m | noarr::get_length<'a'>(idx<'c'>(lit<1>))) == 10'013 % 16);

	REQUIRE((m | noarr::offset<'a', 'y', 'b', 'c'>(10, 3333, 500, lit<0>)) == (10 + 500*16 + 3333*10'013L) * sizeof(float));
	REQUIRE((m | noarr::offset<'a', 'y', 'b', 'c'>(10, 3333,   1, lit<1>)) == (10 + 10'000 + 3333*10'013L) * sizeof(float));
}

TEST_CASE("Blocks with static border reused as minor", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'013>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_static<'x', 'c', 'b', 'x'>(16);

	REQUIRE(decltype(m)::signature::all_accept<'y'>);
	REQUIRE(decltype(m)::signature::all_accept<'c'>);
	REQUIRE(decltype(m)::signature::all_accept<'b'>);
	REQUIRE(decltype(m)::signature::all_accept<'x'>);

	REQUIRE((m | noarr::get_length<'y'>()) == 20'000);
	REQUIRE((m | noarr::get_length<'c'>()) == 2);
	REQUIRE((m | noarr::get_length<'b'>(idx<'c'>(lit<0>))) == 10'013 / 16);
	REQUIRE((m | noarr::get_length<'b'>(idx<'c'>(lit<1>))) ==           1);
	REQUIRE((m | noarr::get_length<'x'>(idx<'c'>(lit<0>))) ==          16);
	REQUIRE((m | noarr::get_length<'x'>(idx<'c'>(lit<1>))) == 10'013 % 16);

	REQUIRE((m | noarr::offset<'x', 'y', 'b', 'c'>(10, 3333, 500, lit<0>)) == (10 + 500*16 + 3333*10'013L) * sizeof(float));
	REQUIRE((m | noarr::offset<'x', 'y', 'b', 'c'>(10, 3333,   1, lit<1>)) == (10 + 10'000 + 3333*10'013L) * sizeof(float));
}

TEST_CASE("Blocks with static border reused as major", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'013>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_static<'x', 'c', 'x', 'a'>(16);

	REQUIRE(decltype(m)::signature::all_accept<'y'>);
	REQUIRE(decltype(m)::signature::all_accept<'c'>);
	REQUIRE(decltype(m)::signature::all_accept<'x'>);
	REQUIRE(decltype(m)::signature::all_accept<'a'>);

	REQUIRE((m | noarr::get_length<'y'>()) == 20'000);
	REQUIRE((m | noarr::get_length<'c'>()) == 2);
	REQUIRE((m | noarr::get_length<'x'>(idx<'c'>(lit<0>))) == 10'013 / 16);
	REQUIRE((m | noarr::get_length<'x'>(idx<'c'>(lit<1>))) ==           1);
	REQUIRE((m | noarr::get_length<'a'>(idx<'c'>(lit<0>))) ==          16);
	REQUIRE((m | noarr::get_length<'a'>(idx<'c'>(lit<1>))) == 10'013 % 16);

	REQUIRE((m | noarr::offset<'a', 'y', 'x', 'c'>(10, 3333, 500, lit<0>)) == (10 + 500*16 + 3333*10'013L) * sizeof(float));
	REQUIRE((m | noarr::offset<'a', 'y', 'x', 'c'>(10, 3333,   1, lit<1>)) == (10 + 10'000 + 3333*10'013L) * sizeof(float));
}

TEST_CASE("Blocks with static border reused as is-border", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'013>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_static<'x', 'x', 'b', 'a'>(16);

	REQUIRE(decltype(m)::signature::all_accept<'y'>);
	REQUIRE(decltype(m)::signature::all_accept<'x'>);
	REQUIRE(decltype(m)::signature::all_accept<'b'>);
	REQUIRE(decltype(m)::signature::all_accept<'a'>);

	REQUIRE((m | noarr::get_length<'y'>()) == 20'000);
	REQUIRE((m | noarr::get_length<'x'>()) == 2);
	REQUIRE((m | noarr::get_length<'b'>(idx<'x'>(lit<0>))) == 10'013 / 16);
	REQUIRE((m | noarr::get_length<'b'>(idx<'x'>(lit<1>))) ==           1);
	REQUIRE((m | noarr::get_length<'a'>(idx<'x'>(lit<0>))) ==          16);
	REQUIRE((m | noarr::get_length<'a'>(idx<'x'>(lit<1>))) == 10'013 % 16);

	REQUIRE((m | noarr::offset<'a', 'y', 'b', 'x'>(10, 3333, 500, lit<0>)) == (10 + 500*16 + 3333*10'013L) * sizeof(float));
	REQUIRE((m | noarr::offset<'a', 'y', 'b', 'x'>(10, 3333,   1, lit<1>)) == (10 + 10'000 + 3333*10'013L) * sizeof(float));
}

TEST_CASE("Blocks with static border traverser", "[blocks traverser]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 11>()
		^ noarr::array<'y', 5>()
		^ noarr::into_blocks_static<'x', 'c', 'b', 'a'>(4);

	std::size_t i = 0, a = 0, b = 0, c = 0, y = 0;

	noarr::traverser(m).for_each([&](auto s){
		REQUIRE(noarr::get_index<'y'>(s) == y);
		REQUIRE(noarr::get_index<'c'>(s) == c);
		REQUIRE(noarr::get_index<'b'>(s) == b);
		REQUIRE(noarr::get_index<'a'>(s) == a);

		if constexpr(decltype(noarr::get_index<'c'>(s))::value == 0) { // blocked body
			if(++a == 4) {
				a = 0;
				if(++b == 2) {
					b = 0;
					++c;
				}
			}
		} else if constexpr(decltype(noarr::get_index<'c'>(s))::value == 1) { // border
			if(++a == 3) {
				a = 0;
				c = 0;
				++y;
			}
		} else { // should not happen
			REQUIRE(false);
		}

		REQUIRE((m | noarr::offset(s)) == i * sizeof(float));
		i++;
	});

	REQUIRE(a == 0);
	REQUIRE(b == 0);
	REQUIRE(c == 0);
	REQUIRE(y == 5);
	REQUIRE(i == 11 * 5);
}
