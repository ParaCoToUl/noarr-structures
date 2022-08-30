#include <catch2/catch.hpp>

#include <array>
#include <iostream>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/traverser.hpp"
#include "noarr/structures/reorder.hpp"

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
