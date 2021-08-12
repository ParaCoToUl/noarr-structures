#ifndef NOARR_MATRIX_FUNCTIONS_HPP
#define NOARR_MATRIX_FUNCTIONS_HPP

#include <cassert>

#include "noarr/structures_extended.hpp"

// read IMPORTANT from matrix.cpp first

/**
 * @brief Takes 2 noarr matrices and multiplyes them.
 *
 * @tparam matrix1: First noarr matrix
 * @tparam matrix2: Second noarr matrix
 * @tparam structure: Structure defining structure to be used by result noarr matrix
 * @return noarr::bag<Structure> noarr matrix created from source noarr matrices
 */
template<typename Structure1, typename Structure2, typename Structure3>
noarr::bag<Structure3> noarr_matrix_multiply(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2, Structure3 structure)
{
	auto result = noarr::make_bag(structure);

	int x1_size = matrix1.template get_length<'n'>();
	int y1_size = matrix1.template get_length<'m'>();
	int x2_size = matrix2.template get_length<'n'>();
	int y2_size = matrix2.template get_length<'m'>();

	assert(x1_size == y2_size);

	for (int i = 0; i < x2_size; i++)
		for (int j = 0; j < y1_size; j++)
		{
			int sum = 0;

			for (int k = 0; k < x1_size; k++)
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
 * @return noarr::bag<Structure> noarr matrix created from source noarr matrix
 */
template<typename Structure1, typename Structure2>
void noarr_matrix_copy(noarr::bag<Structure1>& source, Structure2 structure)
{
	auto result = noarr::make_bag(structure);

	for (int i = 0; i < source.template get_length<'n'>(); i++)
		for (int j = 0; j < source.template get_length<'m'>(); j++)
			result.template at<'n', 'm'>(i, j) = source.template at<'n', 'm'>(i, j);
}

/**
 * @brief Takes noarr matrix and transposes it.
 *
 * @tparam matrix: source noarr matrix
 */
template<typename Structure>
void noarr_matrix_transpose(noarr::bag<Structure>& matrix1)
{
	int x_size = matrix1.template get_length<'n'>();
	int y_size = matrix1.template get_length<'m'>();

	assert(x_size == y_size);

	for (int i = 0; i < x_size; i++)
		for (int j = i; j < y_size; j++)
		{
			int& value1 = matrix1.template at<'n', 'm'>(i, j);
			int& value2 = matrix1.template at<'n', 'm'>(j, i);

			std::swap(value1, value2);
		}
}

/**
 * @brief Takes noarr matrix and multiplies dit by scalar. It takes noarr matrix and scalar to multiply with matrix.
 *
 * @tparam matrix: source noarr matrix
 * @param scalar: scalar
 */
template<typename Structure>
void noarr_matrix_scalar_multiplication(noarr::bag<Structure>& matrix, int scalar)
{
	int x_size = matrix.template get_length<'n'>();
	int y_size = matrix.template get_length<'m'>();

	for (int i = 0; i < x_size; i++)
		for (int j = i; j < y_size; j++)
			matrix.template at<'n', 'm'>(i, j) *= scalar;
}

#endif // NOARR_MATRIX_FUNCTIONS_HPP
