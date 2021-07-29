#ifndef NOARR_MATRIX_FUNCTIONS_HPP
#define NOARR_MATRIX_FUNCTIONS_HPP

#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"
#include "noarr/structures/wrapper.hpp"
#include "noarr/structures/bag.hpp"

template<typename Structure1, typename Structure2>
void matrix_copy(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2)
{
	int x_size = matrix1.template get_length<'x'>();
	int y_size = matrix1.template get_length<'y'>();

	REQUIRE(x_size == matrix2.template get_length<'x'>());
	REQUIRE(y_size == matrix2.template get_length<'y'>());

	for (int i = 0; i < x_size; i++)
		for (int j = 0; j < y_size; j++)
		{
			int& value2 = matrix2.at<'x', 'y'>(i, j);
			value2 = matrix1.at<'x', 'y'>(i, j);
		}
}

template<typename Structure>
void matrix_transpose(noarr::bag<Structure>& matrix1)
{
	int x_size = matrix1.template get_length<'x'>();
	int y_size = matrix1.template get_length<'y'>();

	REQUIRE(x_size == y_size);

	for (int i = 0; i < x_size; i++)
		for (int j = i; j < y_size; j++)
		{
			int& value1 = matrix1.at<'x', 'y'>(i, j);
			int& value2 = matrix1.at<'x', 'y'>(j, i);
			std::swap(value1, value2);
		}
}

template<typename Structure1, typename Structure2, typename Structure3>
void matrix_add(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2, noarr::bag<Structure3>& matrix3)
{
	int x_size = matrix1.template get_length<'x'>();
	int y_size = matrix1.template get_length<'y'>();

	REQUIRE(x_size == matrix2.template get_length<'x'>());
	REQUIRE(y_size == matrix2.template get_length<'y'>());
	REQUIRE(x_size == matrix3.template get_length<'x'>());
	REQUIRE(y_size == matrix3.template get_length<'y'>());

	for (int i = 0; i < x_size; i++)
		for (int j = 0; j < y_size; j++)
		{
			int& value1 = matrix1.structure().at<'x', 'y'>(i, j);
			int& value2 = matrix2.structure().at<'x', 'y'>(i, j);
			int& value3 = matrix3.structure().at<'x', 'y'>(i, j);
			value3 = value1 + value2;
		}
}

template<typename Structure>
void matrix_scalar_multiplication(noarr::bag<Structure>& matrix1, int scalar)
{
	int x_size = matrix1.template get_length<'x'>();
	int y_size = matrix1.template get_length<'y'>();

	for (int i = 0; i < x_size; i++)
		for (int j = i; j < y_size; j++)
			matrix1.structure().at<'x', 'y'>(i, j) *= scalar;
}

template<typename Structure1, typename Structure2, typename Structure3>
void matrix_multiply(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2, noarr::bag<Structure3>& matrix3)
{
	int x1_size = matrix1.template get_length<'x'>();
	int y1_size = matrix1.template get_length<'y'>();
	int x2_size = matrix2.template get_length<'x'>();
	int y2_size = matrix2.template get_length<'y'>();
	int x3_size = matrix3.template get_length<'x'>();
	int y3_size = matrix3.template get_length<'y'>();

	assert(x1_size == y2_size);
	assert(y1_size == y3_size);
	assert(x2_size == x3_size);

	for (int i = 0; i < x3_size; i++)
	{
		for (int j = 0; j < y3_size; j++)
		{
			int sum = 0;

			for (int k = 0; k < x1_size; k++)
			{
				int& value1 = matrix1.structure().at<'x', 'y'>(k, j);
				int& value2 = matrix2.structure().at<'x', 'y'>(i, k);
				sum += value1 * value2;
			}

			matrix3.structure().at<'x', 'y'>(i, j) = sum;
		}
	}
}
#endif // NOARR_MATRIX_FUNCTIONS_HPP