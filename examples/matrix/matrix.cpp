#include <iostream>
#include <array>
#include <utility>
#include <vector>
#include <cassert>
#include <tuple>

#include "z_curve.hpp"
#include "noarr_matrix_functions.hpp"

using matrix_rows = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
using matrix_columns = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
using matrix_zcurve = noarr::z_curve<'x', 'y', noarr::sized_vector<'a', noarr::scalar<int>>>;

struct matrix
{
	matrix(int X, int Y, std::vector<int>&& Ary) : x(X), y(Y), ary(std::move(Ary)) {};
	matrix(int X, int Y, std::vector<int>& Ary) : x(X), y(Y), ary(Ary) {};

	int x;
	int y;
	std::vector<int> ary;

	int& at(int x_, int y_) { return ary[x_ + y_ * x]; }
	const int& at(int x_, int y_) const { return ary[x_ + y_ * x]; }

	void print()
	{
		for (int i = 0; i < x; i++)
		{
			for (int j = 0; j < y; j++)
				std::cout << at(i, j) << " ";

			std::cout << std::endl;
		}

		std::cout << std::endl;
	}
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
			m.at(i, j) = matrix1.template at<'x', 'y'>(i, j);

	return m;
}

template<typename Structure>
void clasic_matrix_to_noarr(matrix& m1, noarr::bag<Structure>& matrix1)
{
	int x_size = matrix1.structure().template get_length<'x'>();
	int y_size = matrix1.structure().template get_length<'y'>();

	for (int i = 0; i < x_size; i++)
		for (int j = 0; j < y_size; j++)
			matrix1.template at<'x', 'y'>(i, j) = m1.at(i, j);
}

void clasic_matrix_multiply(matrix& m1, matrix& m2, matrix& m3)
{
	int x1_size = m1.x;
	int y1_size = m1.y;
	int x2_size = m2.x;
	int y2_size = m2.y;
	int x3_size = m3.x;
	int y3_size = m3.y;

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
				int& value1 = m1.at(k, j);
				int& value2 = m2.at(i, k);

				sum += value1 * value2;
			}

			m3.at(i, j) = sum;
		}
	}
}

template<typename Structure>
void matrix_demo(int size, Structure structure)
{
	matrix m1 = get_clasic_matrix(size, size);
	matrix m2 = get_clasic_matrix(size, size);
	matrix m3 = get_clasic_matrix(size, size);

	std::cout << "Matrix 1:" << std::endl;
	m1.print();

	std::cout << "Matrix 2:" << std::endl;
	m2.print();

	auto n1 = noarr::bag(structure);
	auto n2 = noarr::bag(structure);
	auto n3 = noarr::bag(structure);

	clasic_matrix_to_noarr(m1, n1);
	clasic_matrix_to_noarr(m2, n2);

	clasic_matrix_multiply(m1, m2, m3);
	matrix_multiply(n1, n2, n3);

	matrix m4 = noarr_matrix_to_clasic(n3);

	std::cout << "Classic multiplication:" << std::endl;
	m3.print();

	std::cout << "Noarr multiplication:" << std::endl;
	m4.print();

	assert(are_equal_matrices(m3, m4));
}

int main()
{
	while (true)
	{
		std::cout << "Please select matrix layout to be used:" << std::endl;
		std::cout << "1 - rows" << std::endl;
		std::cout << "2 - columns" << std::endl;
		std::cout << "3 - zcurve (the size has to be a power of 2)" << std::endl;
		std::cout << "4 - exit programm" << std::endl;

		int layout;

		std::cin >> layout;

		if (layout == 4)
			return 0;

		std::cout << "Please select the size of the matrix (for simplicity, only square matrices are supported):" << std::endl;

		int size;

		std::cin >> size;

		if (size < 1)
			return -1;

		if (layout == 1)
			matrix_demo(size, matrix_rows() | noarr::set_length<'x'>(size) | noarr::set_length<'y'>(size));
		else if (layout == 2)
			matrix_demo(size, matrix_columns() | noarr::set_length<'x'>(size) | noarr::set_length<'y'>(size));
		else if (layout == 3)
			matrix_demo(size, matrix_zcurve(noarr::sized_vector<'a', noarr::scalar<int>>(noarr::scalar<int>(), size * size), noarr::helpers::z_curve_bottom<'x'>(size), noarr::helpers::z_curve_bottom<'y'>(size)));
	}
}
