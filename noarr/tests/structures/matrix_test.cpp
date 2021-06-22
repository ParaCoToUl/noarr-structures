#include <catch2/catch.hpp>
//#include "noarr/structures.hpp"

#include <iostream>
#include <array>

#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"
#include "noarr/structures/wrapper.hpp"

#include <tuple>

template<typename Structure>
struct Bag
{
private:
	noarr::wrapper<Structure> layout_;
	std::unique_ptr<char[]> data_;

public:
	explicit Bag(Structure s)
		: layout_(noarr::wrap(s)),
		data_(std::make_unique<char[]>(layout().get_size())) { }

	constexpr const noarr::wrapper<Structure> &layout() const noexcept { return layout_; }
	
	constexpr char *data() const noexcept { return data_.get(); }

	void clear()
	{
		auto size_ = layout().get_size();
		for (std::size_t i = 0; i < size_; i++)
			data_[i] = 0;
	}
};

template<typename Structure>
auto GetBag(Structure s)
{
	return Bag<Structure>(s);
}

template<typename Structure>
auto GetBag(noarr::wrapper<Structure> s)
{
	return Bag<Structure>(s.unwrap());
}

enum class MatrixDataLayout { Rows = 1, Columns = 2, Zcurve = 3 };
struct RowsStruct {};
struct ColumnsStruct {};
struct ZcurveStruct {};

template<MatrixDataLayout layout>
struct GetMatrixStructreStructure;

template<>
struct GetMatrixStructreStructure<MatrixDataLayout::Rows>
{
	static constexpr auto GetImageStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<>
struct GetMatrixStructreStructure<MatrixDataLayout::Columns>
{
	static constexpr auto GetImageStructure()
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







template<typename Structure>
void matrixCopy(Bag<Structure> matrix1, Bag<Structure> matrix2)
{
	int x_size = matrix1.layout().template get_length<'x'>();
	int y_size = matrix1.layout().template get_length<'y'>();

	REQUIRE(x_size == matrix2.layout().template get_length<'x'>());
	REQUIRE(y_size == matrix2.layout().template get_length<'y'>());

	for (int i = 0; i < x_size; i++)
	{
		for (int j = 0; j < y_size; j++)
		{
			int value1 = matrix1.layout().template get_at<'x', 'y'>(matrix1.data(), i, j);
			int& value2 = matrix2.layout().template get_at<'x', 'y'>(matrix2.data(), i, j);
			value2 = value1;
		}
	}
}

template<typename Structure>
void matrixTranspose(Bag<Structure> matrix1)
{
	int x_size = matrix1.layout().template get_length<'x'>();
	int y_size = matrix1.layout().template get_length<'y'>();

	REQUIRE(x_size == y_size);

	for (int i = 0; i < x_size; i++)
	{
		for (int j = 0; j < y_size; j++)
		{
			int value1 = matrix1.layout().template get_at<'x', 'y'>(matrix1.data(), i, j);
			int& value2 = matrix1.layout().template get_at<'x', 'y'>(matrix1.data(), j, i);
			value2 = value1;
		}
	}
}

template<typename Structure, MatrixDataLayout layout>
auto matrixClone(Bag<Structure> matrix1)
{
	int x_size = matrix1.layout().template get_length<'x'>();
	int y_size = matrix1.layout().template get_length<'y'>();

	auto matrix2 = GetBag(noarr::wrap(GetMatrixStructreStructure<layout>::GetImageStructure()).template set_length<'x'>(x_size).template set_length<'y'>(y_size));

	matrixCopy(matrix1, matrix2);

	return matrix2;
}

/*template<typename Structure>
auto matrixClone(Bag<Structure> matrix1, MatrixDataLayout layout)
{
	if (layout == MatrixDataLayout.Rows)
		return matrixClone<MatrixDataLayout.Rows>(matrix1);
	else if (layout == MatrixDataLayout.Columns)
		return matrixClone<MatrixDataLayout.Columns>(matrix1);
	else if (layout == MatrixDataLayout.Zcurve)
		return matrixClone<MatrixDataLayout.Zcurve>(matrix1);
}*/

template<typename Structure>
auto matrixClone(Bag<Structure> matrix1, RowsStruct)
{
	return matrixClone<MatrixDataLayout::Rows>(matrix1);
}

template<typename Structure>
auto matrixClone(Bag<Structure> matrix1, ColumnsStruct)
{
	return matrixClone<MatrixDataLayout::Columns>(matrix1);
}

template<typename Structure>
auto matrixClone(Bag<Structure> matrix1, ZcurveStruct)
{
	return matrixClone<MatrixDataLayout::Zcurve>(matrix1);
}



template<MatrixDataLayout layout, std::size_t width, std::size_t height, std::size_t pixel_range = 256>
void histogram_template_test()
{
	auto image = GetBag(noarr::wrap(GetMatrixStructreStructure<layout>::GetImageStructure()).template set_length<'x'>(width).template set_length<'y'>(height));
	CHECK(image.layout().get_size() == width * height * sizeof(int));

	int y_size = image.layout().template get_length<'y'>();
	CHECK(y_size == height);

	auto histogram = GetBag(noarr::array<'x', pixel_range, noarr::scalar<int>>());
	CHECK(histogram.layout().get_size() == pixel_range * sizeof(int));

	image.clear();
	histogram.clear();

	int x_size = image.layout().template get_length<'x'>();
	REQUIRE(x_size == width);

	y_size = image.layout().template get_length<'y'>();
	REQUIRE(y_size == height);

	for (int i = 0; i < x_size; i++)
	{
		for (int j = 0; j < y_size; j++)
		{
			//int& pixel_value = *((int*)(image.blob + x_fixed.fix<'y'>(j).offset())); // v1
			//int& pixel_value = *((int*)x_fixed.fix<'y'>(j).get_at(image.blob)); // v2
			int pixel_value = image.layout().template get_at<'x','y'>(image.data(), i, j); // v3

			if (pixel_value != 0)
				FAIL();

			int& histogram_value = histogram.layout().template get_at<'x'>(histogram.data(), pixel_value);
			histogram_value = histogram_value + 1;
		}
	}
}

TEST_CASE("Matrix prototype 720 x 480 with 16 colors", "[Matrix prototype]")
{
	histogram_template_test<MatrixDataLayout::Rows, 720, 480, 16>();
}

TEST_CASE("Matrix prototype 720 x 480", "[Matrix prototype]")
{
	histogram_template_test<MatrixDataLayout::Rows, 720, 480>();
}

TEST_CASE("Matrix prototype 1080 x 720", "[Matrix prototype]")
{
	histogram_template_test<MatrixDataLayout::Rows, 1080, 720>();
}

TEST_CASE("Matrix prototype 1920 x 1080", "[Matrix prototype]")
{
	histogram_template_test<MatrixDataLayout::Rows, 1920, 1080>();
}

template<MatrixDataLayout layout, std::size_t width, std::size_t height, std::size_t pixel_range = 256>
void histogram_template_test_clear()
{
	auto image = GetBag(noarr::wrap(GetMatrixStructreStructure<layout>::GetImageStructure()).template set_length<'x'>(width).template set_length<'y'>(height)); // image size 
	auto histogram = GetBag(noarr::array<'x', pixel_range, noarr::scalar<int>>()); // lets say that every image has 256 pixel_range

	image.clear();
	histogram.clear();

	int x_size = image.layout().template get_length<'x'>();
	int y_size = image.layout().template get_length<'y'>();

	for (int i = 0; i < x_size; i++)
	{
		for (int j = 0; j < y_size; j++)
		{
			int pixel_value = image.layout().template get_at<'x', 'y'>(image.data(), i, j);

			int& histogram_value = histogram.layout().template get_at<'x'>(histogram.data(), pixel_value);
			histogram_value = histogram_value + 1;
		}
	}
}
