/**
 * @file compile_test.cpp
 * @brief This file contains the code snippets from the root README.md ensuring they are correct
 * 
 */

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/iterator.hpp>

TEST_CASE("Iterator trivial", "[iterator]") {
	auto array = noarr::array<'x', 20, noarr::scalar<int>>();

	auto data = std::array<int, 20>();

	auto range = array | noarr::iterate<'x'>();
	
	int i = 0;

	for(auto it = range.begin(); it != range.end(); ++it, ++i) {
		array | noarr::get_at(data.data(), *it) = i;
	}

	auto it = range.begin();
	bool consistent = true;
	bool consistent_length = true;

	for (std::size_t x = 0; x != 20; ++x) {
		consistent_length = it != range.end();
		consistent = consistent_length && x == noarr::get_index<'x'>(*it) && (array | noarr::get_at(data.data(), *it)) == int(x);
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
	
	int i = 0;

	for(auto it = range.begin(); it != range.end(); ++it, ++i) {
		array | noarr::get_at(data.data(), *it) = i;
	}

	auto it = range.begin();
	bool consistent = true;
	bool consistent_length = true;

	for (std::size_t x = 0; x != 20; ++x) {
		for(std::size_t y = 0; y != 30; ++y) {
			consistent_length = it != range.end();
			consistent = consistent_length && x == noarr::get_index<'x'>(*it) && y == noarr::get_index<'y'>(*it) && (array | noarr::get_at(data.data(), *it)) == int(x * 30 + y);
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
	
	int i = 0;

	for(auto it = range.begin(); it != range.end(); ++it, ++i) {
		array | noarr::get_at(data.data(), *it) = i;
	}

	auto it = range.begin();
	bool consistent = true;
	bool consistent_length = true;

	for(std::size_t y = 0; y != 30; ++y) {
		for (std::size_t x = 0; x != 20; ++x) {
			consistent_length = it != range.end();
			consistent = consistent_length && x == noarr::get_index<'x'>(*it) && y == noarr::get_index<'y'>(*it) && (array | noarr::get_at(data.data(), *it)) == int(y * 20 + x);
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
