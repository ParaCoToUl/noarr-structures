/**
 * @file compile_test.cpp
 * @brief This file contains the code snippets from the root README.
 * @details The code snippets are extended with checks to ensure the code compiles and runs as expected.
 *
 */

#include <noarr_test/macros.hpp>

#include <cstddef>

#include <utility>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/bag.hpp>

namespace {
	constexpr std::size_t ROWS = 10;
	constexpr std::size_t COLS = 20;
} // namespace


TEST_CASE("Examples for Noarr Structures", "[compile test]") {
	const auto row_major_matrix = noarr::scalar<int>() ^
	                              noarr::array<'r', ROWS>() ^
	                              noarr::array<'c', COLS>();

	const auto col_major_matrix = noarr::scalar<int>() ^
	                              noarr::array<'c', COLS>() ^
	                              noarr::array<'r', ROWS>();


	auto matrix = noarr::bag(row_major_matrix);
	const std::size_t row = 2, col = 3;

	int& value = matrix[noarr::idx<'r', 'c'>(row, col)];
	value = 42;

	const auto offset = (matrix | noarr::offset(noarr::idx<'r', 'c'>(row, col))) / sizeof(int);

	REQUIRE(static_cast<std::ptrdiff_t>(offset) == &value - static_cast<int*>(matrix.data()));

	{
		auto matrix2 = noarr::bag(col_major_matrix, matrix.data());
		// equivalent row and col
		const std::size_t row = offset / COLS, col = offset % COLS;

		const int& value = matrix2[noarr::idx<'r', 'c'>(row, col)];

		REQUIRE(value == 42);
	}

	const auto snd_matrix = matrix.get_ref();

	[matrix = std::move(matrix), snd_matrix](){
		(void)matrix;
		(void)snd_matrix;
	}();

}

#include <noarr/traversers.hpp>

TEST_CASE("Examples for Noarr Traversers", "[compile tests]") {
	const auto row_major_matrix = noarr::scalar<int>() ^
	                              noarr::array<'r', ROWS>() ^
	                              noarr::array<'c', COLS>();

	const auto matrix = noarr::bag(row_major_matrix);

	const auto traverser = noarr::traverser(matrix);

	std::size_t iteration = 0;

	traverser | [&](const auto idx) {
		const auto [row, col] = noarr::get_indices<'r', 'c'>(idx);
		matrix[idx] = static_cast<int>(col * ROWS + row);

		REQUIRE(iteration == col * ROWS + row);
		iteration++;
	};

	{
		const auto traverser = noarr::traverser(matrix) ^ noarr::hoist<'r', 'c'>();

		iteration = 0;

		traverser | [&](const auto idx) {
			const auto [row, col] = noarr::get_indices<'r', 'c'>(idx);

			const int value = matrix[idx];

			REQUIRE(value == static_cast<int>(col * ROWS + row));
			REQUIRE(iteration == row * COLS + col);
			iteration++;
		};
	}
}
