#include <catch2/catch.hpp>

#include <tuple>

#include "noarr/structures_extended.hpp"


enum MatrixDataLayout { Rows = 0, Columns = 1 };

template<MatrixDataLayout layout>
struct MatrixStructureGetter;

template<>
struct MatrixStructureGetter<MatrixDataLayout::Rows>
{
	static constexpr auto GetMatrixStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<>
struct MatrixStructureGetter<MatrixDataLayout::Columns>
{
	static constexpr auto GetMatrixStructure()
	{
		return noarr::vector<'y', noarr::vector<'x', noarr::scalar<int>>>();
	}
};

template<typename Matrix1, typename Matrix2>
void matrix_copy(Matrix1& matrix1, Matrix2& matrix2)
{
	std::size_t x_size = matrix1.template get_length<'x'>();
	std::size_t y_size = matrix1.template get_length<'y'>();

	REQUIRE(x_size == matrix2.template get_length<'x'>());
	REQUIRE(y_size == matrix2.template get_length<'y'>());

	for (std::size_t i = 0; i < x_size; i++)
		for (std::size_t j = 0; j < y_size; j++)
			matrix2.template at<'x', 'y'>(i, j) = matrix1.template at<'x', 'y'>(i, j);
}

template<typename Matrix1>
void matrix_transpose(Matrix1& matrix1)
{
	std::size_t x_size = matrix1.template get_length<'x'>();
	std::size_t y_size = matrix1.template get_length<'y'>();

	REQUIRE(x_size == y_size);

	for (std::size_t i = 0; i < x_size; i++)
		for (std::size_t j = i; j < y_size; j++)
			std::swap(matrix1.at<'x', 'y'>(i, j), matrix1.at<'x', 'y'>(j, i));
}

template<typename Matrix1, typename Matrix2, typename Matrix3>
void matrix_add(Matrix1& matrix1, Matrix2& matrix2, Matrix3& matrix3)
{
	std::size_t x_size = matrix1.template get_length<'x'>();
	std::size_t y_size = matrix1.template get_length<'y'>();

	REQUIRE(x_size == matrix2.template get_length<'x'>());
	REQUIRE(y_size == matrix2.template get_length<'y'>());
	REQUIRE(x_size == matrix3.template get_length<'x'>());
	REQUIRE(y_size == matrix3.template get_length<'y'>());

	for (std::size_t i = 0; i < x_size; i++)
		for (std::size_t j = 0; j < y_size; j++)
		{
			int& value1 = matrix1.template at<'x', 'y'>(i, j);
			int& value2 = matrix2.template at<'x', 'y'>(i, j);
			int& value3 = matrix3.template at<'x', 'y'>(i, j);

			value3 = value1 + value2;
		}
}

template<typename Matrix1>
void matrix_scalar_multiplication(Matrix1& matrix1, int scalar)
{
	std::size_t x_size = matrix1.template get_length<'x'>();
	std::size_t y_size = matrix1.template get_length<'y'>();

	for (std::size_t i = 0; i < x_size; i++)
		for (std::size_t j = i; j < y_size; j++)
			matrix1.template at<'x', 'y'>(i, j) *= scalar;
}

template<typename Matrix1, typename Matrix2, typename Matrix3>
void matrix_multiply(Matrix1& matrix1, Matrix2& matrix2, Matrix3& matrix3)
{
	std::size_t x1_size = matrix1.template get_length<'x'>();
	std::size_t y1_size = matrix1.template get_length<'y'>();
	std::size_t x2_size = matrix2.template get_length<'x'>();
	std::size_t y2_size = matrix2.template get_length<'y'>();
	std::size_t x3_size = matrix3.template get_length<'x'>();
	std::size_t y3_size = matrix3.template get_length<'y'>();

	REQUIRE(x1_size == y2_size);
	REQUIRE(y1_size == y3_size);
	REQUIRE(x2_size == x3_size);

	for (std::size_t i = 0; i < x3_size; i++)
	{
		for (std::size_t j = 0; j < y3_size; j++)
		{
			int sum = 0;

			for (std::size_t k = 0; k < x1_size; k++)
			{
				int& value1 = matrix1.template at<'x', 'y'>(k, j);
				int& value2 = matrix2.template at<'x', 'y'>(i, k);

				sum += value1 * value2;
			}

			matrix3.template at<'x', 'y'>(i, j) = sum;
		}
	}
}

template<MatrixDataLayout layout>
void matrix_template_test(std::size_t size)
{
	// using different kinds of bags
	auto m1 = noarr::make_vector_bag(noarr::wrap(MatrixStructureGetter<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
	auto m2 = noarr::make_vector_bag(noarr::wrap(MatrixStructureGetter<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
	auto m3 = noarr::make_bag(noarr::wrap(MatrixStructureGetter<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));

	matrix_multiply(m1, m2, m3);
}

void matrix_template_test_runtime(MatrixDataLayout layout, std::size_t size)
{
	if (layout == MatrixDataLayout::Rows)
		matrix_template_test<MatrixDataLayout::Rows>(size);
	else if (layout == MatrixDataLayout::Columns)
		matrix_template_test<MatrixDataLayout::Columns>(size);
}


struct matrix
{
	matrix(std::size_t X, std::size_t Y, std::vector<int>&& Ary) : x(X), y(Y), ary(std::move(Ary)) {};

	std::size_t x;
	std::size_t y;
	std::vector<int> ary;

	int& at(std::size_t x_, std::size_t y_) { return ary[x_ + y_ * x]; }
	const int& at(std::size_t x_, std::size_t y_) const { return ary[x_ + y_ * x]; }
};

matrix get_clasic_matrix(std::size_t x, std::size_t y)
{
	const std::size_t length = x * y;
	std::vector<int> ary;

	for (std::size_t i = 0; i < length; i++)
		ary.push_back(rand() % 10);

	return matrix(x, y, std::move(ary));
}

bool are_equal_matrices(matrix& m1, matrix& m2)
{
	if (m1.x != m2.x)
		return false;

	if (m1.y != m2.y)
		return false;

	const std::size_t length = m1.x * m1.y;
	for (std::size_t i = 0; i < length; i++)
		if (m1.ary[i] != m2.ary[i])
			return false;

	return true;
}

template<typename Matrix1>
matrix noarr_matrix_to_clasic(Matrix1& matrix1)
{
	std::size_t x_size = matrix1.template get_length<'x'>();
	std::size_t y_size = matrix1.template get_length<'y'>();

	matrix m = get_clasic_matrix(x_size, y_size);

	for (std::size_t i = 0; i < x_size; i++)
		for (std::size_t j = 0; j < y_size; j++)
			m.at(i, j) = matrix1.template at<'x', 'y'>(i, j);

	return m;
}

template<typename Matrix1>
void clasic_matrix_to_noarr(matrix& m1, Matrix1& matrix1)
{
	std::size_t x_size = matrix1.template get_length<'x'>();
	std::size_t y_size = matrix1.template get_length<'y'>();

	for (std::size_t i = 0; i < x_size; i++)
		for (std::size_t j = 0; j < y_size; j++)
			matrix1.template at<'x', 'y'>(i, j) = m1.at(i, j);
}

void clasic_matrix_multiply(matrix& m1, matrix& m2, matrix& m3)
{
	std::size_t x1_size = m1.x;
	std::size_t y1_size = m1.y;
	std::size_t x2_size = m2.x;
	std::size_t y2_size = m2.y;
	std::size_t x3_size = m3.x;
	std::size_t y3_size = m3.y;

	REQUIRE(x1_size == y2_size);
	REQUIRE(y1_size == y3_size);
	REQUIRE(x2_size == x3_size);

	for (std::size_t i = 0; i < x3_size; i++)
	{
		for (std::size_t j = 0; j < y3_size; j++)
		{
			int sum = 0;

			for (std::size_t k = 0; k < x1_size; k++)
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
void matrix_demo_template(std::size_t size)
{

	matrix m1 = get_clasic_matrix(size, size);
	matrix m2 = get_clasic_matrix(size, size);
	matrix m3 = get_clasic_matrix(size, size);

	auto n1 = noarr::make_bag(noarr::wrap(MatrixStructureGetter<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
	auto n2 = noarr::make_bag(noarr::wrap(MatrixStructureGetter<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
	auto n3 = noarr::make_bag(noarr::wrap(MatrixStructureGetter<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));

	clasic_matrix_to_noarr(m1, n1);
	clasic_matrix_to_noarr(m2, n2);

	clasic_matrix_multiply(m1, m2, m3);
	matrix_multiply(n1, n2, n3);

	matrix m4 = noarr_matrix_to_clasic(n3);

	REQUIRE(are_equal_matrices(m3, m4));
}

void matrix_demo(MatrixDataLayout layout, std::size_t size)
{
	if (layout == MatrixDataLayout::Rows)
		matrix_demo_template<MatrixDataLayout::Rows>(size);
	else if (layout == MatrixDataLayout::Columns)
		matrix_demo_template<MatrixDataLayout::Columns>(size);
}

TEST_CASE("Small matrix demo", "[Small matrix demo]")
{
	matrix_demo(MatrixDataLayout::Rows, 10);
}

TEST_CASE("Small matrix demo 2", "[Small matrix demo 2]")
{
	matrix_demo(MatrixDataLayout::Rows, 20);
}

TEST_CASE("Small matrix multimplication Rows", "[Small matrix multimplication Rows]")
{
	matrix_template_test_runtime(MatrixDataLayout::Rows, 10);
}

TEST_CASE("Small matrix multimplication Columns", "[Small matrix multimplication Columns]")
{
	matrix_template_test_runtime(MatrixDataLayout::Columns, 10);
}
