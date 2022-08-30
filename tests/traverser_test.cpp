#include <catch2/catch.hpp>

#include <array>
#include <iostream>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/traverser.hpp"
#include "noarr/structures/reorder.hpp"
#include "noarr/structures/shortcuts.hpp"

using namespace noarr;

TEST_CASE("Traverser trivial", "[traverser]") {
	using at = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array<'y', 30, noarr::array<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array<'x', 20, noarr::array<'z', 40, noarr::scalar<int>>>;

	using xt = noarr::array<'z', 40, noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>>;

	using u1t = union_t<at>;
	using u2t = union_t<at, bt>;
	using u3t = union_t<at, bt, ct>;

	at a;
	bt b;
	ct c;

	u1t u1;
	u2t u2;
	u3t u3;

	REQUIRE(u1.length<'x'>(state<>()) == 20);
	REQUIRE(u1.length<'y'>(state<>()) == 30);

	REQUIRE(u2.length<'x'>(state<>()) == 20);
	REQUIRE(u2.length<'y'>(state<>()) == 30);
	REQUIRE(u2.length<'z'>(state<>()) == 40);

	REQUIRE(u3.length<'x'>(state<>()) == 20);
	REQUIRE(u3.length<'y'>(state<>()) == 30);
	REQUIRE(u3.length<'z'>(state<>()) == 40);

	REQUIRE(std::is_same_v<u1t::signature, at::signature>);
	REQUIRE(std::is_same_v<u2t::signature, xt::signature>);
	REQUIRE(std::is_same_v<u3t::signature, xt::signature>);

	int i = 0;

	traverser(a, b, c).for_each([&i](auto sa, auto sb, auto sc){
		static_assert( decltype(sa)::template contains<index_in<'x'>>);
		static_assert( decltype(sa)::template contains<index_in<'y'>>);
		static_assert(!decltype(sa)::template contains<index_in<'z'>>);

		static_assert(!decltype(sb)::template contains<index_in<'x'>>);
		static_assert( decltype(sb)::template contains<index_in<'y'>>);
		static_assert( decltype(sb)::template contains<index_in<'z'>>);

		static_assert( decltype(sc)::template contains<index_in<'x'>>);
		static_assert(!decltype(sc)::template contains<index_in<'y'>>);
		static_assert( decltype(sc)::template contains<index_in<'z'>>);

		REQUIRE(sa.template get<index_in<'x'>>() == sc.template get<index_in<'x'>>());
		REQUIRE(sa.template get<index_in<'y'>>() == sb.template get<index_in<'y'>>());
		REQUIRE(sb.template get<index_in<'z'>>() == sc.template get<index_in<'z'>>());

		REQUIRE(sa.template get<index_in<'x'>>() < 20);
		REQUIRE(sb.template get<index_in<'y'>>() < 30);
		REQUIRE(sc.template get<index_in<'z'>>() < 40);

		int j =
			+ sc.template get<index_in<'z'>>() * 30 * 20
			+ sa.template get<index_in<'x'>>() * 30
			+ sb.template get<index_in<'y'>>()
		;

		REQUIRE(i == j);

		i++;
	});

	REQUIRE(i == 20*30*40);
}

TEST_CASE("Traverser ordered", "[traverser]") {
	using at = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array<'y', 30, noarr::array<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array<'x', 20, noarr::array<'z', 40, noarr::scalar<int>>>;

	using xt = noarr::array<'z', 40, noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>>;

	using u1t = union_t<at>;
	using u2t = union_t<at, bt>;
	using u3t = union_t<at, bt, ct>;

	at a;
	bt b;
	ct c;

	u1t u1;
	u2t u2;
	u3t u3;

	REQUIRE(u1.length<'x'>(state<>()) == 20);
	REQUIRE(u1.length<'y'>(state<>()) == 30);

	REQUIRE(u2.length<'x'>(state<>()) == 20);
	REQUIRE(u2.length<'y'>(state<>()) == 30);
	REQUIRE(u2.length<'z'>(state<>()) == 40);

	REQUIRE(u3.length<'x'>(state<>()) == 20);
	REQUIRE(u3.length<'y'>(state<>()) == 30);
	REQUIRE(u3.length<'z'>(state<>()) == 40);

	REQUIRE(std::is_same_v<u1t::signature, at::signature>);
	REQUIRE(std::is_same_v<u2t::signature, xt::signature>);
	REQUIRE(std::is_same_v<u3t::signature, xt::signature>);

	int i = 0;

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>()).for_each([&i](auto sa, auto sb, auto sc){
		static_assert( decltype(sa)::template contains<index_in<'x'>>);
		static_assert( decltype(sa)::template contains<index_in<'y'>>);
		static_assert(!decltype(sa)::template contains<index_in<'z'>>);

		static_assert(!decltype(sb)::template contains<index_in<'x'>>);
		static_assert( decltype(sb)::template contains<index_in<'y'>>);
		static_assert( decltype(sb)::template contains<index_in<'z'>>);

		static_assert( decltype(sc)::template contains<index_in<'x'>>);
		static_assert(!decltype(sc)::template contains<index_in<'y'>>);
		static_assert( decltype(sc)::template contains<index_in<'z'>>);

		REQUIRE(sa.template get<index_in<'x'>>() == sc.template get<index_in<'x'>>());
		REQUIRE(sa.template get<index_in<'y'>>() == sb.template get<index_in<'y'>>());
		REQUIRE(sb.template get<index_in<'z'>>() == sc.template get<index_in<'z'>>());

		REQUIRE(sa.template get<index_in<'x'>>() < 20);
		REQUIRE(sb.template get<index_in<'y'>>() < 30);
		REQUIRE(sc.template get<index_in<'z'>>() < 40);

		int j =
			+ sa.template get<index_in<'x'>>() * 30 * 40
			+ sb.template get<index_in<'y'>>() * 40
			+ sc.template get<index_in<'z'>>()
		;

		REQUIRE(i == j);

		i++;
	});

	REQUIRE(i == 20*30*40);
}

TEST_CASE("Traverser ordered renamed", "[traverser]") {
	using at = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array<'y', 30, noarr::array<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array<'x', 20, noarr::array<'z', 40, noarr::scalar<int>>>;

	using xt = noarr::array<'z', 40, noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>>;

	using u1t = union_t<at>;
	using u2t = union_t<at, bt>;
	using u3t = union_t<at, bt, ct>;

	at a;
	bt b;
	ct c;

	u1t u1;
	u2t u2;
	u3t u3;

	REQUIRE(u1.length<'x'>(state<>()) == 20);
	REQUIRE(u1.length<'y'>(state<>()) == 30);

	REQUIRE(u2.length<'x'>(state<>()) == 20);
	REQUIRE(u2.length<'y'>(state<>()) == 30);
	REQUIRE(u2.length<'z'>(state<>()) == 40);

	REQUIRE(u3.length<'x'>(state<>()) == 20);
	REQUIRE(u3.length<'y'>(state<>()) == 30);
	REQUIRE(u3.length<'z'>(state<>()) == 40);

	REQUIRE(std::is_same_v<u1t::signature, at::signature>);
	REQUIRE(std::is_same_v<u2t::signature, xt::signature>);
	REQUIRE(std::is_same_v<u3t::signature, xt::signature>);

	int i = 0;

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>() ^ noarr::rename<'y', 't'>()).for_each([&i](auto sa, auto sb, auto sc){
		static_assert( decltype(sa)::template contains<index_in<'x'>>);
		static_assert( decltype(sa)::template contains<index_in<'y'>>);
		static_assert(!decltype(sa)::template contains<index_in<'z'>>);

		static_assert(!decltype(sb)::template contains<index_in<'x'>>);
		static_assert( decltype(sb)::template contains<index_in<'y'>>);
		static_assert( decltype(sb)::template contains<index_in<'z'>>);

		static_assert( decltype(sc)::template contains<index_in<'x'>>);
		static_assert(!decltype(sc)::template contains<index_in<'y'>>);
		static_assert( decltype(sc)::template contains<index_in<'z'>>);
		
		REQUIRE(sa.template get<index_in<'x'>>() == sc.template get<index_in<'x'>>());
		REQUIRE(sa.template get<index_in<'y'>>() == sb.template get<index_in<'y'>>());
		REQUIRE(sb.template get<index_in<'z'>>() == sc.template get<index_in<'z'>>());

		REQUIRE(sa.template get<index_in<'x'>>() < 20);
		REQUIRE(sb.template get<index_in<'y'>>() < 30);
		REQUIRE(sc.template get<index_in<'z'>>() < 40);

		int j =
			+ sa.template get<index_in<'x'>>() * 30 * 40
			+ sb.template get<index_in<'y'>>() * 40
			+ sc.template get<index_in<'z'>>()
		;

		REQUIRE(i == j);

		i++;
	});

	REQUIRE(i == 20*30*40);
}

TEST_CASE("Traverser ordered renamed access", "[traverser]") {
	using at = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array<'y', 30, noarr::array<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array<'x', 20, noarr::array<'z', 40, noarr::scalar<int>>>;

	int ad[20*30];
	int bd[30*40];
	int cd[20*40];

	void* ap = ad;
	void* bp = bd;
	void* cp = cd;

	at a;
	bt b;
	ct c;

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>() ^ noarr::rename<'y', 't'>()).for_each([&](auto sa, auto sb, auto sc){
		REQUIRE((a | offset(sa)) / sizeof(int) == sa.template get<index_in<'x'>>() * 30 + sa.template get<index_in<'y'>>());
		REQUIRE((b | offset(sb)) / sizeof(int) == sb.template get<index_in<'y'>>() * 40 + sb.template get<index_in<'z'>>());
		REQUIRE((c | offset(sc)) / sizeof(int) == sc.template get<index_in<'x'>>() * 40 + sc.template get<index_in<'z'>>());

		REQUIRE((char*) &(a | get_at(ap, sa)) == (char*) ap + (a | offset(sa)));
		REQUIRE((char*) &(b | get_at(bp, sb)) == (char*) bp + (b | offset(sb)));
		REQUIRE((char*) &(c | get_at(cp, sc)) == (char*) cp + (c | offset(sc)));
	});
}

TEST_CASE("Traverser ordered renamed access blocked", "[traverser blocks]") {
	using at = noarr::array<'y', 30, noarr::array<'x', 20, noarr::scalar<int>>>;
	using bt = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	at a;
	bt b;

	std::size_t i = 0;

	traverser(a, b).order(noarr::into_blocks<'x', 'u', 'v'>(4)).for_each([&](auto sa, auto sb){
		REQUIRE(!decltype(sa)::template contains<index_in<'u'>> & !decltype(sa)::template contains<index_in<'v'>>);
		REQUIRE(!decltype(sb)::template contains<index_in<'u'>> & !decltype(sb)::template contains<index_in<'v'>>);
		REQUIRE( decltype(sa)::template contains<index_in<'x'>> &  decltype(sa)::template contains<index_in<'y'>>);
		REQUIRE( decltype(sb)::template contains<index_in<'x'>> &  decltype(sb)::template contains<index_in<'y'>>);

		REQUIRE(sa.template get<index_in<'x'>>() == sb.template get<index_in<'x'>>());
		REQUIRE(sa.template get<index_in<'y'>>() == sb.template get<index_in<'y'>>());

		std::size_t x = sa.template get<index_in<'x'>>();
		std::size_t y = sa.template get<index_in<'y'>>();
		REQUIRE(y*20 + x == i);
		i++;
	});

	REQUIRE(i == 20*30);
}

TEST_CASE("Traverser ordered renamed access strip mined", "[traverser shortcuts blocks]") {
	using at = noarr::array<'y', 30, noarr::array<'x', 20, noarr::scalar<int>>>;
	using bt = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>;

	at a;
	bt b;

	std::size_t i = 0;

	traverser(a, b).order(noarr::strip_mine<'x', 'u', 'v'>(4)).for_each([&](auto sa, auto sb){
		REQUIRE(!decltype(sa)::template contains<index_in<'u'>> & !decltype(sa)::template contains<index_in<'v'>>);
		REQUIRE(!decltype(sb)::template contains<index_in<'u'>> & !decltype(sb)::template contains<index_in<'v'>>);
		REQUIRE( decltype(sa)::template contains<index_in<'x'>> &  decltype(sa)::template contains<index_in<'y'>>);
		REQUIRE( decltype(sb)::template contains<index_in<'x'>> &  decltype(sb)::template contains<index_in<'y'>>);

		REQUIRE(sa.template get<index_in<'x'>>() == sb.template get<index_in<'x'>>());
		REQUIRE(sa.template get<index_in<'y'>>() == sb.template get<index_in<'y'>>());

		std::size_t x = sa.template get<index_in<'x'>>();
		std::size_t y = sa.template get<index_in<'y'>>();
		std::size_t u = x / 4;
		std::size_t v = x % 4;
		REQUIRE(u*30*4 + y*4 + v == i);
		i++;
	});

	REQUIRE(i == 20*30);
}
