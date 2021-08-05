#include <iostream>
#include <array>
#include <utility>
#include <vector>
#include <cassert>
#include <tuple>

// raw c++ matrix implementation is from now on a "classic matrix"
// matrix implementation in noarr will be referecend as "noarr matrix"
// whole example assumes int matrices

// definion of z-curve data stricture
#include "z_curve.hpp"
// definitions of basic matrix functions: matrix multiplication, scalar multiplication, copy, matrix transpose and matrix add
#include "noarr_matrix_functions.hpp"

using matrix_rows = noarr::vector<'n', noarr::vector<'m', noarr::scalar<int>>>;
using matrix_columns = noarr::vector<'n', noarr::vector<'m', noarr::scalar<int>>>;
using matrix_zcurve = noarr::z_curve<'n', 'm', noarr::sized_vector<'a', noarr::scalar<int>>>;

// raw c++ structure, which implements matrix ("classic matrix")
struct classic_matrix
{
	// constructors
	classic_matrix(int X, int Y, std::vector<int>&& Ary) : n(X), m(Y), ary(std::move(Ary)) {};
	classic_matrix(int X, int Y, std::vector<int>& Ary) : n(X), m(Y), ary(Ary) {};

	// width
	int n;
	// heigth
	int m;
	// data vector (flattened matrix by rows into std::vector<int>)
	std::vector<int> ary;

	// element access functions (returns reference into data vector based on input indexes)
	int& at(int n_, int m_) { return ary[n_ + m_ * n]; }
	const int& at(int n_, int m_) const { return ary[n_ + m_ * n]; }

	// printing function which prints the whole matrix into the standard output
	void print()
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
				std::cout << at(i, j) << " ";

			std::cout << std::endl;
		}

		std::cout << std::endl;
	}
};

// function returng random classic matrix with values in range [0 to 9] with size n x m. 
classic_matrix get_clasic_matrix(int n, int m)
{
	// data container inicialization
	const int length = n * m;
	std::vector<int> ary;

	// random values generation
	for (int i = 0; i < length; i++)
		ary.push_back(rand() % 10);

	// matrix construction
	return classic_matrix(n, m, std::move(ary));
}

// function which compares two classic matrices for value equality
bool are_equal_classic_matrices(classic_matrix& m1, classic_matrix& m2)
{
	// n size must be the same
	if (m1.n != m2.n)
		return false;

	// m size must be the same
	if (m1.m != m2.m)
		return false;

	// data container has to be the same size, we will compare values for equality
	const int length = m1.n * m1.m;
	for (int i = 0; i < length; i++)
		if (m1.ary[i] != m2.ary[i])
			return false;

	// if all tests were passed up until now the matrices have to be value-equal
	return true;
}

// function converting noarr matrix to classic matrix
// it takes source noarr matrix as argument
template<typename Structure>
classic_matrix noarr_matrix_to_clasic(noarr::bag<Structure>& source)
{
	// we will cache matrix size values
	int n_size = source.structure().template get_length<'n'>();
	int m_size = source.structure().template get_length<'m'>();

	// we will allocate target classic matrix
	classic_matrix target = get_clasic_matrix(n_size, m_size);

	// we will go through the matrix and copy noarr matrix into a classic matrix
	for (int i = 0; i < n_size; i++)
		for (int j = 0; j < m_size; j++)
			target.at(i, j) = source.template at<'n', 'm'>(i, j);

	return target;
}

// function converting classic matrix to noarr matrix
// it takes source classic matrix and target noarr matrix as arguments
template<typename Structure>
void clasic_matrix_to_noarr(classic_matrix& source, noarr::bag<Structure>& target)
{
	// we will cache matrix size values
	int n_size = target.structure().template get_length<'n'>();
	int m_size = target.structure().template get_length<'m'>();

	// we will fo through matrix and copy classic matrix into noarr matrix
	for (int i = 0; i < n_size; i++)
		for (int j = 0; j < m_size; j++)
			target.template at<'n', 'm'>(i, j) = source.at(i, j);
}

// function multiplying classic matrices
// it takes 2 source classic matrices and returns multiplied matrix
classic_matrix clasic_matrix_multiply(classic_matrix& matrix1, classic_matrix& matrix2)
{
	// we will cache matrix size values
	int n1_size = matrix1.n;
	int m1_size = matrix1.m;
	int n2_size = matrix2.n;
	int m2_size = matrix2.m;

	// n1 and m2 have to be equal
	assert(n1_size == m2_size);

	// we will allocate target classic matrix
	classic_matrix output = get_clasic_matrix(n2_size, m1_size);

	// standart matrix multiplication
	for (int i = 0; i < n2_size; i++)
	{
		for (int j = 0; j < m1_size; j++)
		{
			int sum = 0;

			for (int k = 0; k < n1_size; k++)
				sum += matrix1.at(k, j) * matrix2.at(i, k);

			output.at(i, j) = sum;
		}
	}

	return output;
}


template<typename Structure>
void matrix_demo(int size, Structure structure)
{
	classic_matrix m1 = get_clasic_matrix(size, size);
	std::cout << "Matrix 1:" << std::endl;
	m1.print();

	classic_matrix m2 = get_clasic_matrix(size, size);
	std::cout << "Matrix 2:" << std::endl;
	m2.print();

	auto n1 = noarr::bag(structure);
	auto n2 = noarr::bag(structure);
	auto n3 = noarr::bag(structure);

	clasic_matrix_to_noarr(m1, n1);
	clasic_matrix_to_noarr(m2, n2);

	classic_matrix m3 = clasic_matrix_multiply(m1, m2);
	std::cout << "Classic multiplication:" << std::endl;
	m3.print();

	noarr_matrix_multiply(n1, n2, n3);
	classic_matrix m4 = noarr_matrix_to_clasic(n3);


	std::cout << "Noarr multiplication:" << std::endl;
	m4.print();

	assert(are_equal_classic_matrices(m3, m4));
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
			matrix_demo(size, matrix_rows() | noarr::set_length<'n'>(size) | noarr::set_length<'m'>(size));
		else if (layout == 2)
			matrix_demo(size, matrix_columns() | noarr::set_length<'n'>(size) | noarr::set_length<'m'>(size));
		else if (layout == 3)
			matrix_demo(size, matrix_zcurve(noarr::sized_vector<'a', noarr::scalar<int>>(noarr::scalar<int>(), size * size), noarr::helpers::z_curve_bottom<'n'>(size), noarr::helpers::z_curve_bottom<'m'>(size)));
	}
}
