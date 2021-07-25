#include <catch2/catch.hpp>
//#include "noarr/structures.hpp"

#include <iostream>
#include <array>
#include <utility>

#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"
#include "noarr/structures/wrapper.hpp"
#include "noarr/structures/bag.hpp"

#include <tuple>

enum MatrixDataLayout { Rows = 0, Columns = 1, Zcurve = 2 };

template<MatrixDataLayout layout>
struct GetMatrixStructreStructure;

template<>
struct GetMatrixStructreStructure<MatrixDataLayout::Rows>
{
	static constexpr auto GetMatrixStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<>
struct GetMatrixStructreStructure<MatrixDataLayout::Columns>
{
	static constexpr auto GetMatrixStructure()
	{
		return noarr::vector<'y', noarr::vector<'x', noarr::scalar<int>>>();
	}
};

template<>
struct GetMatrixStructreStructure<MatrixDataLayout::Zcurve>
{
	static constexpr auto GetMatrixStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<typename Structure1, typename Structure2>
void matrix_copy(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2)
{
	int x_size = matrix1.structure().template get_length<'x'>();
	int y_size = matrix1.structure().template get_length<'y'>();

	REQUIRE(x_size == matrix2.structure().template get_length<'x'>());
	REQUIRE(y_size == matrix2.structure().template get_length<'y'>());

	for (int i = 0; i < x_size; i++)
		for (int j = 0; j < y_size; j++)
		{
			int& value2 = matrix2.structure().template get_at<'x', 'y'>(matrix2.data(), i, j);
			value2 = matrix1.structure().template get_at<'x', 'y'>(matrix1.data(), i, j);
		}
}

template<typename Structure>
void matrix_transpose(noarr::bag<Structure>& matrix1)
{
	int x_size = matrix1.structure().template get_length<'x'>();
	int y_size = matrix1.structure().template get_length<'y'>();

	REQUIRE(x_size == y_size);

	for (int i = 0; i < x_size; i++)
		for (int j = i; j < y_size; j++)
		{
			int& value1 = matrix1.structure().template get_at<'x', 'y'>(matrix1.data(), i, j);
			int& value2 = matrix1.structure().template get_at<'x', 'y'>(matrix1.data(), j, i);
			std::swap(value1, value2);
		}
}

template<typename Structure1, typename Structure2, typename Structure3>
void matrix_add(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2, noarr::bag<Structure3>& matrix3)
{
	int x_size = matrix1.structure().template get_length<'x'>();
	int y_size = matrix1.structure().template get_length<'y'>();

	REQUIRE(x_size == matrix2.structure().template get_length<'x'>());
	REQUIRE(y_size == matrix2.structure().template get_length<'y'>());
	REQUIRE(x_size == matrix3.structure().template get_length<'x'>());
	REQUIRE(y_size == matrix3.structure().template get_length<'y'>());

	for (int i = 0; i < x_size; i++)
		for (int j = 0; j < y_size; j++)
		{
			int& value1 = matrix1.structure().template get_at<'x', 'y'>(matrix1.data(), i, j);
			int& value2 = matrix2.structure().template get_at<'x', 'y'>(matrix2.data(), i, j);
			int& value3 = matrix3.structure().template get_at<'x', 'y'>(matrix3.data(), i, j);
			value3 = value1 + value2;
		}
}

template<typename Structure>
void matrix_scalar_multiplication(noarr::bag<Structure>& matrix1, int scalar)
{
	int x_size = matrix1.structure().template get_length<'x'>();
	int y_size = matrix1.structure().template get_length<'y'>();

	for (int i = 0; i < x_size; i++)
		for (int j = i; j < y_size; j++)
			matrix1.structure().template get_at<'x', 'y'>(matrix1.data(), i, j) *= scalar;
}

template<typename Structure1, typename Structure2, typename Structure3>
void matrix_multiply(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2, noarr::bag<Structure3>& matrix3)
{
	int x1_size = matrix1.structure().template get_length<'x'>();
	int y1_size = matrix1.structure().template get_length<'y'>();
	int x2_size = matrix2.structure().template get_length<'x'>();
	int y2_size = matrix2.structure().template get_length<'y'>();
	int x3_size = matrix3.structure().template get_length<'x'>();
	int y3_size = matrix3.structure().template get_length<'y'>();

	REQUIRE(x1_size == y2_size);
	REQUIRE(y1_size == y3_size);
	REQUIRE(x2_size == x3_size);

	for (int i = 0; i < x3_size; i++)
	{
		for (int j = 0; j < y3_size; j++)
		{
			int sum = 0;

			for (int k = 0; k < x1_size; k++)
			{
				int& value1 = matrix1.structure().template get_at<'x', 'y'>(matrix1.data(), k, j);
				int& value2 = matrix2.structure().template get_at<'x', 'y'>(matrix2.data(), i, k);
				sum += value1 * value2;
			}

			matrix3.structure().template get_at<'x', 'y'>(matrix3.data(), i, j) = sum;
		}
	}
}

template<MatrixDataLayout layout>
void matrix_template_test(int size)
{
	auto m1 = noarr::bag(noarr::wrap(GetMatrixStructreStructure<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
	auto m2 = noarr::bag(noarr::wrap(GetMatrixStructreStructure<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
	auto m3 = noarr::bag(noarr::wrap(GetMatrixStructreStructure<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));

	matrix_multiply(m1, m2, m3);
}

void matrix_template_test_runtime(MatrixDataLayout layout, int size)
{
	if (layout == MatrixDataLayout::Rows)
		matrix_template_test<MatrixDataLayout::Rows>(size);
	else if (layout == MatrixDataLayout::Columns)
		matrix_template_test<MatrixDataLayout::Columns>(size);
	else if (layout == MatrixDataLayout::Zcurve)
		matrix_template_test<MatrixDataLayout::Zcurve>(size);
}


struct matrix
{
	matrix(int X, int Y, std::vector<int>&& Ary) : x(X), y(Y), ary(std::move(Ary)) {};

	int x;
	int y;
	std::vector<int> ary;

	int& at(int x_, int y_) { return ary[x_ + y_ * x]; }
	const int& at(int x_, int y_) const { return ary[x_ + y_ * x]; }
};

matrix get_clasic_matrix(int x, int y)
{
	const int length = x * y;
	std::vector<int> ary;
	for (int i = 0; i < length; i++)
		ary.push_back(rand() % 10);

	return matrix(x, y, std::move(ary));
}

bool are_equal_matrices(matrix& m1, matrix& m2)
{
	if (m1.x != m2.x)
		return false;
	if (m1.y != m2.y)
		return false;

	const int length = m1.x * m1.y;
	for (int i = 0; i < length; i++)
		if (m1.ary[i] != m2.ary[i])
			return false;

	return true;
}

template<typename Structure>
matrix noarr_matrix_to_clasic(noarr::bag<Structure>& matrix1)
{
	int x_size = matrix1.structure().template get_length<'x'>();
	int y_size = matrix1.structure().template get_length<'y'>();

	matrix m = get_clasic_matrix(x_size, y_size);

	for (int i = 0; i < x_size; i++)
		for (int j = 0; j < y_size; j++)
			m.at(i, j) = matrix1.structure().template get_at<'x', 'y'>(matrix1.data(), i, j);

	return m;
}

template<typename Structure>
void clasic_matrix_to_naorr(matrix& m1, noarr::bag<Structure>& matrix1)
{
	int x_size = matrix1.structure().template get_length<'x'>();
	int y_size = matrix1.structure().template get_length<'y'>();

	for (int i = 0; i < x_size; i++)
		for (int j = 0; j < y_size; j++)
			matrix1.structure().template get_at<'x', 'y'>(matrix1.data(), i, j) = m1.at(i, j);
}

void clasic_matrix_multiply(matrix& m1, matrix& m2, matrix& m3)
{
	int x1_size = m1.x;
	int y1_size = m1.y;
	int x2_size = m2.x;
	int y2_size = m2.y;
	int x3_size = m3.x;
	int y3_size = m3.y;

	REQUIRE(x1_size == y2_size);
	REQUIRE(y1_size == y3_size);
	REQUIRE(x2_size == x3_size);

	for (int i = 0; i < x3_size; i++)
	{
		for (int j = 0; j < y3_size; j++)
		{
			int sum = 0;

			for (int k = 0; k < x1_size; k++)
			{
				int& value1 = m1.at(k, j);
				int& value2 = m2.at(i, k);
				sum += value1 * value2;
			}

			m3.at(i, j) = sum;
		}
	}
}

template<MatrixDataLayout layout>
void matrix_demo_template(int size)
{

	matrix m1 = get_clasic_matrix(size, size);
	matrix m2 = get_clasic_matrix(size, size);
	matrix m3 = get_clasic_matrix(size, size);

	auto n1 = noarr::bag(noarr::wrap(GetMatrixStructreStructure<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
	auto n2 = noarr::bag(noarr::wrap(GetMatrixStructreStructure<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
	auto n3 = noarr::bag(noarr::wrap(GetMatrixStructreStructure<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));

	clasic_matrix_to_naorr(m1, n1);
	clasic_matrix_to_naorr(m2, n2);

	clasic_matrix_multiply(m1, m2, m3);
	matrix_multiply(n1, n2, n3);

	matrix m4 = noarr_matrix_to_clasic(n3);

	REQUIRE(are_equal_matrices(m3, m4));
}

void matrix_demo(MatrixDataLayout layout, int size)
{
	if (layout == MatrixDataLayout::Rows)
		matrix_demo_template<MatrixDataLayout::Rows>(size);
	else if (layout == MatrixDataLayout::Columns)
		matrix_demo_template<MatrixDataLayout::Columns>(size);
	else if (layout == MatrixDataLayout::Zcurve)
		matrix_demo_template<MatrixDataLayout::Zcurve>(size);
}
