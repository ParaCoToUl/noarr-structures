/**
 * @file compile_test.cpp
 * @brief This file contains the code snippets from the root README.md ensuring they are correct
 * 
 */

#include <catch2/catch.hpp>

#include <array>
#include <iostream>

#include "noarr/structures_extended.hpp"

TEST_CASE("Iterator Trivial", "[iterator]") {
	auto array = noarr::array<'x', 20, noarr::scalar<int>>();

	auto data = std::array<int, 20>();

	auto range = array | noarr::iterate<'x'>();
	REQUIRE(std::tuple_size<decltype(range.begin())>::value == 2);
	
	int i = 0;

	for(auto it = range.begin(); it != range.end(); ++it, ++i) {
		auto idx = std::get<0>(it);
		idx | noarr::get_at(data.data()) = i;
	}

	auto it = range.begin();
	bool consistent = true;
	bool consistent_length = true;

	for (std::size_t x = 0; x != 20; ++x) {
		auto idx = std::get<0>(it);

		consistent_length = it != range.end();
		consistent = consistent_length && x == std::get<1>(it) && (idx | noarr::get_at(data.data())) == int(x);
		if (!consistent) {
			goto consistency_evaluation;
		}

		++it;
	}

	consistent_length = it == range.end();

consistency_evaluation:
	REQUIRE(consistent_length);
	REQUIRE(consistent);
}

TEST_CASE("Iterator composite", "[iterator]") {
	auto array = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>();

	auto data = std::array<int, 600>();

	auto range = array | noarr::iterate<'x', 'y'>();
	REQUIRE(std::tuple_size<decltype(range.begin())>::value == 3);
	
	int i = 0;

	for(auto it = range.begin(); it != range.end(); ++it, ++i) {
		auto idx = std::get<0>(it);
		idx | noarr::get_at(data.data()) = i;
	}

	auto it = range.begin();
	bool consistent = true;
	bool consistent_length = true;

	for (std::size_t x = 0; x != 20; ++x) {
		for(std::size_t y = 0; y != 30; ++y) {
			auto idx = std::get<0>(it);

			consistent_length = it != range.end();
			consistent = consistent_length && x == std::get<1>(it) && y == std::get<2>(it) && (idx | noarr::get_at(data.data())) == int(x * 30 + y);
			if (!consistent) {
				goto consistency_evaluation;
			}

			++it;
		}
	}

	consistent_length = it == range.end();

consistency_evaluation:
	REQUIRE(consistent_length);
	REQUIRE(consistent);
}

TEST_CASE("Iterator composite reversed", "[iterator]") {
	auto array = noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>>();

	auto data = std::array<int, 600>();

	auto range = array | noarr::iterate<'y', 'x'>();
	REQUIRE(std::tuple_size<decltype(range.begin())>::value == 3);
	
	int i = 0;

	for(auto it = range.begin(); it != range.end(); ++it, ++i) {
		auto idx = std::get<0>(it);
		idx | noarr::get_at(data.data()) = i;
	}

	auto it = range.begin();
	bool consistent = true;
	bool consistent_length = true;

	for(std::size_t y = 0; y != 30; ++y) {
		for (std::size_t x = 0; x != 20; ++x) {
			auto idx = std::get<0>(it);

			consistent_length = it != range.end();
			consistent = consistent_length && y == std::get<1>(it) && x == std::get<2>(it) && (idx | noarr::get_at(data.data())) == int(y * 20 + x);
			if (!consistent) {
				goto consistency_evaluation;
			}

			++it;
		}
	}

	consistent_length = it == range.end();

consistency_evaluation:
	REQUIRE(consistent_length);
	REQUIRE(consistent);
}
