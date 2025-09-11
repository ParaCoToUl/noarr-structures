#ifndef NOARR_MATRIX_FUNCTIONS_HPP
#define NOARR_MATRIX_FUNCTIONS_HPP

#include <cassert>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/interop/bag.hpp"

// read IMPORTANT from matrix.cpp first

/**
 * @brief Takes 2 noarr matrices and multiplyes them.
 *
 * @tparam matrix1: First noarr matrix
 * @tparam matrix2: Second noarr matrix
 * @tparam structure: Structure defining structure to be used by result noarr matrix
 * @return Matrix noarr matrix created from source noarr matrices
 */
template<class Matrix, class Matrix2, class Structure3>
auto noarr_matrix_multiply(Matrix& matrix1, Matrix2& matrix2, Structure3 structure)
{
	auto result = noarr::make_bag(structure);

	std::size_t x1_size = matrix1.template length<'n'>();
	std::size_t y1_size = matrix1.template length<'m'>();
	std::size_t x2_size = matrix2.template length<'n'>();
	std::size_t y2_size = matrix2.template length<'m'>();

	assert(x1_size == y2_size);

	for (std::size_t i = 0; i < x2_size; i++)
		for (std::size_t j = 0; j < y1_size; j++)
		{
			int sum = 0;

			for (std::size_t k = 0; k < x1_size; k++)
			{
				int& value1 = matrix1.template at<'n', 'm'>(k, j);
				int& value2 = matrix2.template at<'n', 'm'>(i, k);

				sum += value1 * value2;
			}

			result.template at<'n', 'm'>(i, j) = sum;
		}

	return result;
}

/**
 * @brief Takes noarr matrix and copies it.
 *
 * @tparam matrix: source noarr matrix
 * @tparam structure: Structure defining structure to be used by result noarr matrix
 * @return Matrix noarr matrix created from source noarr matrix
 */
template<class Matrix, class Structure>
auto noarr_matrix_copy(Matrix& source, Structure structure)
{
	auto result = noarr::make_bag(structure);

	for (std::size_t i = 0; i < source.template length<'n'>(); i++)
		for (std::size_t j = 0; j < source.template length<'m'>(); j++)
			result.template at<'n', 'm'>(i, j) = source.template at<'n', 'm'>(i, j);

	return result;
}

/**
 * @brief Takes noarr matrix and transposes it.
 *
 * @tparam matrix: source noarr matrix
 */
template<class Matrix>
void noarr_matrix_transpose(Matrix& matrix)
{
	std::size_t x_size = matrix.template length<'n'>();
	std::size_t y_size = matrix.template length<'m'>();

	assert(x_size == y_size);

	for (std::size_t i = 0; i < x_size; i++)
		for (std::size_t j = i; j < y_size; j++)
		{
			int& value1 = matrix.template at<'n', 'm'>(i, j);
			int& value2 = matrix.template at<'n', 'm'>(j, i);

			std::swap(value1, value2);
		}
}

/**
 * @brief Takes noarr matrix and multiplies dit by scalar. It takes noarr matrix and scalar to multiply with matrix.
 *
 * @tparam matrix: source noarr matrix
 * @param scalar: scalar
 */
template<class Matrix>
void noarr_matrix_scalar_multiplication(Matrix& matrix, int scalar)
{
	std::size_t x_size = matrix.template length<'n'>();
	std::size_t y_size = matrix.template length<'m'>();

	for (std::size_t i = 0; i < x_size; i++)
		for (std::size_t j = 0; j < y_size; j++)
			matrix.template at<'n', 'm'>(i, j) *= scalar;
}

#endif // NOARR_MATRIX_FUNCTIONS_HPP
