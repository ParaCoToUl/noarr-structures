#include <catch2/catch.hpp>

#include <array>
#include <iostream>

#include "noarr/structures_extended.hpp"

TEST_CASE("One-dimensional array zip", "[zip]") {
	auto a = noarr::array<'x', 3, noarr::scalar<int>>();
	auto b = noarr::array<'x', 3, noarr::scalar<int>>();

	auto zip = noarr::zip<'x'>()(a, b);

	auto iter = zip.begin();
	auto end = zip.end();

	for(std::size_t x = 0; x < 3; x++) {
		REQUIRE(iter != end);
		REQUIRE((*iter).get<'x'>() == x);
		++iter;
	}

	REQUIRE(iter == end);
}

TEST_CASE("Two-dimensional array zip", "[zip]") {
	auto a = noarr::array<'y', 5, noarr::array<'x', 3, noarr::scalar<int>>>();
	auto b = noarr::array<'y', 5, noarr::array<'x', 3, noarr::scalar<int>>>();

	auto zip = noarr::zip<'x', 'y'>()(a, b);

	auto iter = zip.begin();
	auto end = zip.end();

	for(std::size_t y = 0; y < 5; y++) {
		for(std::size_t x = 0; x < 3; x++) {
			REQUIRE(iter != end);
			REQUIRE((*iter).get<'x'>() == x);
			REQUIRE((*iter).get<'y'>() == y);
			++iter;
		}
	}

	REQUIRE(iter == end);
}

TEST_CASE("Two-dimensional array transposed zip", "[zip]") {
	auto a = noarr::array<'y', 5, noarr::array<'x', 3, noarr::scalar<int>>>();
	auto b = noarr::array<'y', 5, noarr::array<'x', 3, noarr::scalar<int>>>();

	auto zip = noarr::zip<'y', 'x'>()(a, b);

	auto iter = zip.begin();
	auto end = zip.end();

	for(std::size_t x = 0; x < 3; x++) {
		for(std::size_t y = 0; y < 5; y++) {
			REQUIRE(iter != end);
			REQUIRE((*iter).get<'x'>() == x);
			REQUIRE((*iter).get<'y'>() == y);
			++iter;
		}
	}

	REQUIRE(iter == end);
}

TEST_CASE("Two-dimensional transposed array zip", "[zip]") {
	auto a = noarr::array<'y', 5, noarr::array<'x', 3, noarr::scalar<int>>>();
	auto b = noarr::array<'x', 3, noarr::array<'y', 5, noarr::scalar<int>>>();

	auto zip = noarr::zip<'x', 'y'>()(a, b);

	auto iter = zip.begin();
	auto end = zip.end();

	for(std::size_t y = 0; y < 5; y++) {
		for(std::size_t x = 0; x < 3; x++) {
			REQUIRE(iter != end);
			REQUIRE((*iter).get<'x'>() == x);
			REQUIRE((*iter).get<'y'>() == y);
			++iter;
		}
	}

	REQUIRE(iter == end);
}

TEST_CASE("Two-dimensional transposed array transposed zip", "[zip]") {
	auto a = noarr::array<'y', 5, noarr::array<'x', 3, noarr::scalar<int>>>();
	auto b = noarr::array<'x', 3, noarr::array<'y', 5, noarr::scalar<int>>>();

	auto zip = noarr::zip<'y', 'x'>()(a, b);

	auto iter = zip.begin();
	auto end = zip.end();

	for(std::size_t x = 0; x < 3; x++) {
		for(std::size_t y = 0; y < 5; y++) {
			REQUIRE(iter != end);
			REQUIRE((*iter).get<'x'>() == x);
			REQUIRE((*iter).get<'y'>() == y);
			++iter;
		}
	}

	REQUIRE(iter == end);
}

TEST_CASE("Scalar zip", "[zip]") {
	auto a = noarr::scalar<int>();
	auto b = noarr::scalar<int>();

	auto zip = noarr::zip<>()(a, b);

	auto iter = zip.begin();
	auto end = zip.end();

	REQUIRE(iter != end);
	// no dimensions to verify
	++iter;

	REQUIRE(iter == end);
}

TEST_CASE("Empty array zip", "[zip]") {
	auto a = noarr::array<'x', 0, noarr::scalar<int>>();
	auto b = noarr::array<'x', 0, noarr::scalar<int>>();

	auto zip = noarr::zip<'x'>()(a, b);

	auto iter = zip.begin();
	auto end = zip.end();

	REQUIRE(iter == end);
}

TEST_CASE("Two-dimensional transposed mixed zip", "[zip]") {
	auto a = noarr::vector<'y', noarr::array<'x', 3, noarr::scalar<int>>>() | noarr::set_length<'y'>(5);
	auto b = noarr::vector<'x', noarr::array<'y', 5, noarr::scalar<int>>>() | noarr::set_length<'x'>(3);

	auto zip = noarr::zip<'x', 'y'>()(a, b);

	auto iter = zip.begin();
	auto end = zip.end();

	for(std::size_t y = 0; y < 5; y++) {
		for(std::size_t x = 0; x < 3; x++) {
			REQUIRE(iter != end);
			REQUIRE((*iter).get<'x'>() == x);
			REQUIRE((*iter).get<'y'>() == y);
			++iter;
		}
	}

	REQUIRE(iter == end);
}

TEST_CASE("Iteration by zip", "[zip]") {
	auto a = noarr::vector<'y', noarr::array<'x', 3, noarr::scalar<int>>>() | noarr::set_length<'y'>(5);
	auto b = noarr::vector<'x', noarr::array<'y', 5, noarr::scalar<int>>>() | noarr::set_length<'x'>(3);

	std::size_t x = 0, y = 0;

	for(auto &&idx : noarr::zip<'x', 'y'>()(a, b)) {
		REQUIRE(idx.get<'x'>() == x);
		REQUIRE(idx.get<'y'>() == y);
		if(++x == 3)
			x = 0, ++y;
	}
}

TEST_CASE("Iteration and indexing by zip", "[zip]") {
	auto a = noarr::array<'y', 5, noarr::array<'x', 3, noarr::scalar<int>>>();
	auto b = noarr::array<'x', 3, noarr::array<'y', 5, noarr::scalar<int>>>();

	auto adata = std::array<int, 5*3>();
	auto bdata = std::array<int, 3*5>();

	for(auto &&idx : noarr::zip<'x', 'y'>()(a, b)) {
		int &aref = a | noarr::get_at(adata.data(), idx);
		int &bref = b | noarr::get_at(bdata.data(), idx);
		auto x = idx.get<'x'>();
		auto y = idx.get<'y'>();
		REQUIRE(&aref == &adata[y*3+x]);
		REQUIRE(&bref == &bdata[x*5+y]);
	}
}
