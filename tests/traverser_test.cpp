#include <noarr_test/macros.hpp>

#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures_extended.hpp>

using namespace noarr;

TEST_CASE("Traverser trivial", "[traverser]") {
	using at = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array_t<'y', 30, noarr::array_t<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array_t<'x', 20, noarr::array_t<'z', 40, noarr::scalar<int>>>;

	using xt = noarr::array_t<'z', 40, noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>>;

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

	STATIC_REQUIRE(std::is_same_v<u1t::signature, at::signature>);
	STATIC_REQUIRE(std::is_same_v<u2t::signature, xt::signature>);
	STATIC_REQUIRE(std::is_same_v<u3t::signature, xt::signature>);

	int i = 0;

	traverser(a, b, c).for_each([&i](auto s){
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);
		REQUIRE(s.template get<index_in<'z'>>() < 40);

		REQUIRE(get_index<'x'>(s) == s.template get<index_in<'x'>>());
		REQUIRE(get_index<'y'>(s) == s.template get<index_in<'y'>>());
		REQUIRE(get_index<'z'>(s) == s.template get<index_in<'z'>>());

		int j =
			+ s.template get<index_in<'z'>>() * 30 * 20
			+ s.template get<index_in<'x'>>() * 30
			+ s.template get<index_in<'y'>>()
		;

		REQUIRE(i == j);

		i++;
	});

	REQUIRE(i == 20*30*40);
}

TEST_CASE("Traverser ordered", "[traverser]") {
	using at = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array_t<'y', 30, noarr::array_t<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array_t<'x', 20, noarr::array_t<'z', 40, noarr::scalar<int>>>;

	using xt = noarr::array_t<'z', 40, noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>>;

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

	STATIC_REQUIRE(std::is_same_v<u1t::signature, at::signature>);
	STATIC_REQUIRE(std::is_same_v<u2t::signature, xt::signature>);
	STATIC_REQUIRE(std::is_same_v<u3t::signature, xt::signature>);

	int i = 0;

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>()).for_each([&i](auto s){
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);
		REQUIRE(s.template get<index_in<'z'>>() < 40);

		int j =
			+ s.template get<index_in<'x'>>() * 30 * 40
			+ s.template get<index_in<'y'>>() * 40
			+ s.template get<index_in<'z'>>()
		;

		REQUIRE(i == j);

		i++;
	});

	REQUIRE(i == 20*30*40);

	traverser(a, b, c) ^ reorder<'x', 'y', 'z'>() | [&i](auto s){
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);
		REQUIRE(s.template get<index_in<'z'>>() < 40);

		int j =
			+ s.template get<index_in<'x'>>() * 30 * 40
			+ s.template get<index_in<'y'>>() * 40
			+ s.template get<index_in<'z'>>()
		;

		REQUIRE(20*30*40 - i == j);

		i--;
	};

	REQUIRE(i == 0);
}

TEST_CASE("Traverser ordered renamed", "[traverser]") {
	using at = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array_t<'y', 30, noarr::array_t<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array_t<'x', 20, noarr::array_t<'z', 40, noarr::scalar<int>>>;

	using xt = noarr::array_t<'z', 40, noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>>;

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

	STATIC_REQUIRE(std::is_same_v<u1t::signature, at::signature>);
	STATIC_REQUIRE(std::is_same_v<u2t::signature, xt::signature>);
	STATIC_REQUIRE(std::is_same_v<u3t::signature, xt::signature>);

	int i = 0;

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>() ^ rename<'y', 't'>()).for_each([&i](auto s){
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);
		REQUIRE(s.template get<index_in<'z'>>() < 40);

		int j =
			+ s.template get<index_in<'x'>>() * 30 * 40
			+ s.template get<index_in<'y'>>() * 40
			+ s.template get<index_in<'z'>>()
		;

		REQUIRE(i == j);

		i++;
	});

	REQUIRE(i == 20*30*40);

	traverser(a, b, c) ^ reorder<'x', 'y', 'z'>() ^ rename<'y', 't'>() | [&i](auto s){
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);
		REQUIRE(s.template get<index_in<'z'>>() < 40);

		int j =
			+ s.template get<index_in<'x'>>() * 30 * 40
			+ s.template get<index_in<'y'>>() * 40
			+ s.template get<index_in<'z'>>()
		;

		REQUIRE(20*30*40 - i == j);

		i--;
	};

	REQUIRE(i == 0);
}

TEST_CASE("Traverser ordered renamed access", "[traverser]") {
	using at = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array_t<'y', 30, noarr::array_t<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array_t<'x', 20, noarr::array_t<'z', 40, noarr::scalar<int>>>;

	int ad[20*30];
	int bd[30*40];
	int cd[20*40];

	void* ap = ad;
	void* bp = bd;
	void* cp = cd;

	at a;
	bt b;
	ct c;

	auto action = [&](auto s){
		REQUIRE((a | offset(s)) / sizeof(int) == s.template get<index_in<'x'>>() * 30 + s.template get<index_in<'y'>>());
		REQUIRE((b | offset(s)) / sizeof(int) == s.template get<index_in<'y'>>() * 40 + s.template get<index_in<'z'>>());
		REQUIRE((c | offset(s)) / sizeof(int) == s.template get<index_in<'x'>>() * 40 + s.template get<index_in<'z'>>());

		REQUIRE(reinterpret_cast<const char*>(&(a | get_at(ap, s))) == static_cast<const char*>(ap) + (a | offset(s)));
		REQUIRE(reinterpret_cast<const char*>(&(b | get_at(bp, s))) == static_cast<const char*>(bp) + (b | offset(s)));
		REQUIRE(reinterpret_cast<const char*>(&(c | get_at(cp, s))) == static_cast<const char*>(cp) + (c | offset(s)));
	};

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>() ^ rename<'y', 't'>()).for_each(action);
	traverser(a, b, c) ^ reorder<'x', 'y', 'z'>() ^ rename<'y', 't'>() | action;
}

TEST_CASE("Traverser ordered renamed access blocked", "[traverser blocks]") {
	using at = noarr::array_t<'y', 30, noarr::array_t<'x', 20, noarr::scalar<int>>>;
	using bt = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

	at a;
	bt b;

	std::size_t i = 0;

	traverser(a, b).order(into_blocks<'x', 'u', 'v'>(4)).for_each([&](auto s){
		STATIC_REQUIRE(!decltype(s)::template contains<index_in<'u'>> & !decltype(s)::template contains<index_in<'v'>>);
		STATIC_REQUIRE( decltype(s)::template contains<index_in<'x'>> &  decltype(s)::template contains<index_in<'y'>>);

		std::size_t x = s.template get<index_in<'x'>>();
		std::size_t y = s.template get<index_in<'y'>>();
		REQUIRE(y*20 + x == i);
		i++;
	});

	REQUIRE(i == 20*30);

	traverser(a, b) ^ into_blocks<'x', 'u', 'v'>(4) | [&](auto s){
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);

		std::size_t x = s.template get<index_in<'x'>>();
		std::size_t y = s.template get<index_in<'y'>>();
		REQUIRE(y*20 + x == 20*30 - i);
		i--;
	};

	REQUIRE(i == 0);
}

TEST_CASE("Traverser ordered renamed access strip mined", "[traverser shortcuts blocks]") {
	using at = noarr::array_t<'y', 30, noarr::array_t<'x', 20, noarr::scalar<int>>>;
	using bt = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;

	at a;
	bt b;

	std::size_t i = 0;

	traverser(a, b).order(strip_mine<'x', 'u', 'v'>(4)).for_each([&](auto s){
		STATIC_REQUIRE(!decltype(s)::template contains<index_in<'u'>> & !decltype(s)::template contains<index_in<'v'>>);
		STATIC_REQUIRE( decltype(s)::template contains<index_in<'x'>> &  decltype(s)::template contains<index_in<'y'>>);

		std::size_t x = s.template get<index_in<'x'>>();
		std::size_t y = s.template get<index_in<'y'>>();
		std::size_t u = x / 4;
		std::size_t v = x % 4;
		REQUIRE(u*30*4 + y*4 + v == i);
		i++;
	});

	REQUIRE(i == 20*30);

	traverser(a, b) ^ strip_mine<'x', 'u', 'v'>(4) | [&i](auto s){
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);

		std::size_t x = s.template get<index_in<'x'>>();
		std::size_t y = s.template get<index_in<'y'>>();
		std::size_t u = x / 4;
		std::size_t v = x % 4;
		REQUIRE(u*30*4 + y*4 + v == 20*30 - i);
		i--;
	};

	REQUIRE(i == 0);
}

TEST_CASE("Traverser sections", "[traverser]") {
	using at = noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>;
	using bt = noarr::array_t<'y', 30, noarr::array_t<'z', 40, noarr::scalar<int>>>;
	using ct = noarr::array_t<'x', 20, noarr::array_t<'z', 40, noarr::scalar<int>>>;

	using xt = noarr::array_t<'z', 40, noarr::array_t<'x', 20, noarr::array_t<'y', 30, noarr::scalar<int>>>>;

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

	STATIC_REQUIRE(std::is_same_v<u1t::signature, at::signature>);
	STATIC_REQUIRE(std::is_same_v<u2t::signature, xt::signature>);
	STATIC_REQUIRE(std::is_same_v<u3t::signature, xt::signature>);

	int i = 0;
	int iters = 0;

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>()).template for_sections<'x', 'y'>([&i, &iters](auto inner){
		auto s = inner.state();

		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);

		int j =
			+ s.template get<index_in<'x'>>() * 30
			+ s.template get<index_in<'y'>>()
		;

		REQUIRE(i == j);

		inner.for_each([&iters, &j](auto s){
			STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

			REQUIRE(s.template get<index_in<'z'>>() < 40);

			int k = j * 40 + s.template get<index_in<'z'>>();

			REQUIRE(iters == k);

			iters++;
		});

		i++;
	});

	REQUIRE(i == 20*30);
	REQUIRE(iters == 20*30*40);

	iters = 0;

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>()).template for_sections<'z', 'x', 'y'>([&iters](auto inner){
		auto s = inner.state();

		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);
		REQUIRE(s.template get<index_in<'z'>>() < 40);

		int j =
			+ s.template get<index_in<'x'>>() * 30 * 40
			+ s.template get<index_in<'y'>>() * 40
			+ s.template get<index_in<'z'>>()
		;

		REQUIRE(iters == j);

		iters++;
	});

	REQUIRE(iters == 20*30*40);

	traverser(a, b, c).order(reorder<'x', 'y', 'z'>()).for_sections([&iters](auto inner){
		auto s = inner.state();

		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);
		REQUIRE(s.template get<index_in<'z'>>() < 40);

		int j =
			+ s.template get<index_in<'x'>>() * 30 * 40
			+ s.template get<index_in<'y'>>() * 40
			+ s.template get<index_in<'z'>>()
		;

		REQUIRE(20*30*40 - iters == j);

		iters--;
	});

	REQUIRE(iters == 0);

	traverser(a, b, c).order(reorder<'y', 'z', 'x'>()).template for_sections<'x', 'y'>([&i, &iters](auto inner){
		auto s = inner.state();

		STATIC_REQUIRE(decltype(s)::template contains<index_in<'x'>>);
		STATIC_REQUIRE(decltype(s)::template contains<index_in<'y'>>);

		REQUIRE(s.template get<index_in<'x'>>() < 20);
		REQUIRE(s.template get<index_in<'y'>>() < 30);

		int j =
			+ s.template get<index_in<'x'>>()
			+ s.template get<index_in<'y'>>() * 20
		;

		REQUIRE(20*30 - i == j);

		inner.for_each([&iters, &j](auto s){
			STATIC_REQUIRE(decltype(s)::template contains<index_in<'z'>>);

			REQUIRE(s.template get<index_in<'z'>>() < 40);

			int k = j * 40 + s.template get<index_in<'z'>>();

			REQUIRE(iters == k);

			iters++;
		});

		i--;
	});

	REQUIRE(i == 0);
	REQUIRE(iters == 20*30*40);
}

TEST_CASE("Traverser update_index (reverse example)", "[traverser shortcuts]") {
	using at = noarr::array_t<'z', 40, noarr::array_t<'y', 30, noarr::array_t<'x', 20, noarr::scalar<int>>>>;

	at a;

	// verify that capture works
	int last_y = 29, last_z = 39;

	traverser(a).for_each([&](auto sa){
		auto sa_ry = update_index<'y'>(sa, [&](auto y){return last_y-y;});
		REQUIRE(sa_ry.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>());
		REQUIRE(sa_ry.template get<index_in<'y'>>() == 29-sa.template get<index_in<'y'>>());
		REQUIRE(sa_ry.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());

		auto sa_rz = update_index<'z'>(sa, [&](auto z){return last_z-z;});
		REQUIRE(sa_rz.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>());
		REQUIRE(sa_rz.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>());
		REQUIRE(sa_rz.template get<index_in<'z'>>() == 39-sa.template get<index_in<'z'>>());
	});
}

TEST_CASE("Traverser neighbor (stencil example)", "[traverser shortcuts]") {
	using at = noarr::array_t<'z', 40, noarr::array_t<'y', 30, noarr::array_t<'x', 20, noarr::scalar<int>>>>;

	at a;

	int i = 0;

	auto method = [&](auto sa){
		auto sa7 = neighbor<'y', 'x'>(sa, -1, -1);
		auto sa8 = neighbor<'y', 'x'>(sa, -1,  0);
		auto sa9 = neighbor<'y', 'x'>(sa, -1, +1);
		auto sa4 = neighbor<'y', 'x'>(sa,  0, -1);
		auto sa6 = neighbor<'y', 'x'>(sa,  0, +1);
		auto sa1 = neighbor<'y', 'x'>(sa, +1, -1);
		auto sa2 = neighbor<'y', 'x'>(sa, +1,  0);
		auto sa3 = neighbor<'y', 'x'>(sa, +1, +1);

		// does not hold generally (`neighbor` could reorder the indices in the state)
		STATIC_REQUIRE(std::is_same_v<decltype(sa7), decltype(sa)>);
		STATIC_REQUIRE(std::is_same_v<decltype(sa8), decltype(sa)>);
		STATIC_REQUIRE(std::is_same_v<decltype(sa9), decltype(sa)>);
		STATIC_REQUIRE(std::is_same_v<decltype(sa4), decltype(sa)>);
		STATIC_REQUIRE(std::is_same_v<decltype(sa6), decltype(sa)>);
		STATIC_REQUIRE(std::is_same_v<decltype(sa1), decltype(sa)>);
		STATIC_REQUIRE(std::is_same_v<decltype(sa2), decltype(sa)>);
		STATIC_REQUIRE(std::is_same_v<decltype(sa3), decltype(sa)>);

		REQUIRE(sa7.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>() - 1);
		REQUIRE(sa8.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>());
		REQUIRE(sa9.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>() + 1);
		REQUIRE(sa4.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>() - 1);
		REQUIRE(sa6.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>() + 1);
		REQUIRE(sa1.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>() - 1);
		REQUIRE(sa2.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>());
		REQUIRE(sa3.template get<index_in<'x'>>() == sa.template get<index_in<'x'>>() + 1);

		REQUIRE(sa7.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>() - 1);
		REQUIRE(sa8.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>() - 1);
		REQUIRE(sa9.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>() - 1);
		REQUIRE(sa4.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>());
		REQUIRE(sa6.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>());
		REQUIRE(sa1.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>() + 1);
		REQUIRE(sa2.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>() + 1);
		REQUIRE(sa3.template get<index_in<'y'>>() == sa.template get<index_in<'y'>>() + 1);

		REQUIRE(sa7.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());
		REQUIRE(sa8.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());
		REQUIRE(sa9.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());
		REQUIRE(sa4.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());
		REQUIRE(sa6.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());
		REQUIRE(sa1.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());
		REQUIRE(sa2.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());
		REQUIRE(sa3.template get<index_in<'z'>>() == sa.template get<index_in<'z'>>());

		REQUIRE(sa7.template get<index_in<'x'>>() < 18);
		REQUIRE(sa7.template get<index_in<'y'>>() < 28);
		REQUIRE(sa3.template get<index_in<'x'>>() >= 2);
		REQUIRE(sa3.template get<index_in<'y'>>() >= 2);
		REQUIRE(sa3.template get<index_in<'x'>>() < 20);
		REQUIRE(sa3.template get<index_in<'y'>>() < 30);

		i++;
	};

	traverser(a).order(slice<'x'>(1, 18) ^ slice<'y'>(1, 28)).for_each(method);

	REQUIRE(i == 40*28*18);

	traverser(a).order(span<'x'>(1, 19) ^ span<'y'>(1, 29)).for_each(method);

	REQUIRE(i == 2 * 40*28*18);
}

TEST_CASE("Traverser step in order", "[traverser shortcuts]") {
	auto a = noarr::scalar<int>() ^ noarr::array<'x', 20>();

	unsigned i = 0;

	traverser(a).order(noarr::step<'x'>(3, 5)).for_each([&](auto sa){
		auto o = a | offset(sa);
		REQUIRE(get_index<'x'>(sa) == (5*i+3));
		REQUIRE(o == (5*i+3) * sizeof(int));
		i++;
	});

	REQUIRE(i == 20/5);
}

TEST_CASE("Traverser step in structure", "[traverser shortcuts]") {
	auto a = noarr::scalar<int>() ^ noarr::array<'x', 20>() ^ noarr::step<'x'>(3, 5);

	unsigned i = 0;

	traverser(a).for_each([&](auto sa){
		auto o = a | offset(sa);
		REQUIRE(get_index<'x'>(sa) == i);
		REQUIRE(o == (5*i+3) * sizeof(int));
		i++;
	});

	REQUIRE(i == 20/5);
}

TEST_CASE("Nested traverser traditional", "[traverser]") {
	auto mat = scalar<float>() ^ array<'j', 50>() ^ array<'i', 50>();

	unsigned ei = 0;

	traverser(mat).order(reorder<'i'>()).for_each([&](auto idxs) {
		auto i = get_index<'i'>(idxs);
		REQUIRE(i == ei);
		auto sliced = mat ^ fix<'i'>(i) ^ slice<'j'>(i, 50 - i);
		unsigned ej = 0;
		traverser(sliced).order(reorder<'j'>()).for_each([&](auto idxs) {
			auto j = get_index<'j'>(idxs);
			REQUIRE(j == ej);
			auto o = sliced | offset(idxs);
			REQUIRE(o == (50*ei + ei+ej) * sizeof(float));
			ej++;
		});
		REQUIRE(ej == 50-ei);
		ei++;
	});

	REQUIRE(ei == 50);
}

TEST_CASE("Nested traverser simplified", "[traverser]") {
	auto mat = scalar<float>() ^ array<'j', 50>() ^ array<'i', 50>();

	unsigned ei = 0;

	traverser(mat).order(reorder<'i'>()).for_each([&](auto idxs) {
		auto i = get_index<'i'>(idxs);
		REQUIRE(i == ei);
		auto sliced = mat ^ fix(idxs) ^ slice<'j'>(i, 50 - i);
		unsigned ej = 0;
		traverser(sliced).for_each([&](auto idxs) {
			auto j = get_index<'j'>(idxs);
			REQUIRE(j == ej);
			auto o = sliced | offset(idxs);
			REQUIRE(o == (50*ei + ei+ej) * sizeof(float));
			ej++;
		});
		REQUIRE(ej == 50-ei);
		ei++;
	});

	REQUIRE(ei == 50);
}

TEST_CASE("Nested traverser for_dims", "[traverser]") {
	auto mat = scalar<float>() ^ array<'j', 50>() ^ array<'i', 50>();

	unsigned ei = 0;

	traverser(mat).for_dims<'i'>([&](auto trav) {
		auto i = get_index<'i'>(trav);
		REQUIRE(i == ei);
		unsigned ej = 0;
		trav.order(shift<'j'>(i)).for_each([&](auto idxs) {
			auto j = get_index<'j'>(idxs);
			REQUIRE(j-i == ej);
			auto o = mat | offset(idxs);
			REQUIRE(o == (50*ei + ei+ej) * sizeof(float));
			ej++;
		});
		REQUIRE(ej == 50-ei);
		ei++;
	});
	REQUIRE(ei == 50);
}

TEST_CASE("Nested traverser for_dims with symmetric_span", "[traverser]") {
	auto mat = scalar<float>() ^ array<'j', 50>() ^ array<'i', 50>();

	unsigned ei = 0;

	traverser(mat).order(span<'i'>(25)).template for_dims<'i'>([&](auto trav) {
		auto i = get_index<'i'>(trav);
		REQUIRE(i == ei);
		unsigned ej = 0;
		trav.order(symmetric_span<'j'>(trav.top_struct(), i)).for_each([&](auto idxs) {
			auto j = get_index<'j'>(idxs);
			REQUIRE(j-i == ej);
			auto o = mat | offset(idxs);
			REQUIRE(o == (50*ei + ei+ej) * sizeof(float));
			ej++;
		});
		REQUIRE(ej == 50-2*ei);
		ei++;
	});
	REQUIRE(ei == 50/2);
}

TEST_CASE("Nested traverser for_dims with symmetric_spans", "[traverser]") {
	auto mat = scalar<float>() ^ array<'j', 50>() ^ array<'i', 50>();

	unsigned ei = 0;

	traverser(mat).order(symmetric_spans<'i', 'j'>(mat, 3, 4)).template for_dims<'i'>([&](auto trav) {
		auto i = get_index<'i'>(trav.state());
		REQUIRE(i == ei + 3);
		unsigned ej = 0;
		trav.for_each([&](auto idxs) {
			auto j = get_index<'j'>(idxs);
			REQUIRE(j == ej + 4);
			auto o = mat | offset(idxs);
			REQUIRE(o == (50*i + j) * sizeof(float));
			ej++;
		});
		REQUIRE(ej == 50-2*4);
		ei++;
	});
	REQUIRE(ei == 50-2*3);
}

TEST_CASE("Traverser single state", "[traverser]") {
	auto t = traverser(scalar<float>() ^ array<'x', 100>()).order(fix<'x'>(42));

	auto s = t.state();

	REQUIRE(get_index<'x'>(s) == 42);
}

TEST_CASE("Traverser single state multi struct", "[traverser]") {
	auto t = traverser(scalar<float>() ^ array<'x', 100>() ^ array<'y', 200>(), scalar<float>() ^ array<'y', 200>() ^ array<'z', 300>()).order(fix<'x'>(42) ^ fix<'y'>(142) ^ fix<'z'>(242));

	auto s = t.state();

	REQUIRE(get_index<'x'>(s) == 42);
	REQUIRE(get_index<'y'>(s) == 142);
	REQUIRE(get_index<'z'>(s) == 242);
}
