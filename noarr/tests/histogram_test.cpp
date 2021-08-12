#include <catch2/catch.hpp>

#include <array>
#include <iostream>
#include <tuple>

#include "noarr/structures_extended.hpp"


enum class ImageDataLayout { ArrayOfArrays = 1, VectorOfVectors = 2, Zcurve = 3 };

template<ImageDataLayout layout>
struct GetImageStructreStructure;

template<>
struct GetImageStructreStructure<ImageDataLayout::ArrayOfArrays>
{
	static constexpr auto GetImageStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<>
struct GetImageStructreStructure<ImageDataLayout::VectorOfVectors>
{
	static constexpr auto GetImageStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<>
struct GetImageStructreStructure<ImageDataLayout::Zcurve>
{
	static constexpr auto GetImageStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<ImageDataLayout layout, std::size_t width, std::size_t height, std::size_t pixel_range = 256>
void histogram_template_test()
{
	auto image = noarr::make_bag(noarr::wrap(GetImageStructreStructure<layout>::GetImageStructure()).template set_length<'x'>(width).template set_length<'y'>(height));

	CHECK(image.get_size() == width * height * sizeof(int));

	auto histogram = noarr::make_bag(noarr::array<'x', pixel_range, noarr::scalar<int>>());
	CHECK(histogram.get_size() == pixel_range * sizeof(int));

	image.clear();
	histogram.clear();

	std::size_t x_size = image.template get_length<'x'>();
	REQUIRE(x_size == width);

	std::size_t y_size = image.template get_length<'y'>();
	REQUIRE(y_size == height);

	for (std::size_t i = 0; i < x_size; i++)
	{
		for (std::size_t j = 0; j < y_size; j++)
		{
			int pixel_value = image.template at<'x','y'>(i, j);

			if (pixel_value != 0)
				FAIL();

			int& histogram_value = histogram.template at<'x'>(pixel_value);
			histogram_value = histogram_value + 1;
		}
	}
}

TEST_CASE("Histogram prototype 128 x 86 with 16 colors", "[Histogram prototype]")
{
	histogram_template_test<ImageDataLayout::ArrayOfArrays, 128, 86, 16>();
}

TEST_CASE("Histogram prototype 128 x 86", "[Histogram prototype]")
{
	histogram_template_test<ImageDataLayout::ArrayOfArrays, 128, 86>();
}

TEST_CASE("Histogram prototype 150 x 86", "[Histogram prototype]")
{
	histogram_template_test<ImageDataLayout::ArrayOfArrays, 150, 86>();
}

TEST_CASE("Histogram prototype 64 x 43", "[Histogram prototype]")
{
	histogram_template_test<ImageDataLayout::ArrayOfArrays, 64, 43>();
}

template<ImageDataLayout layout, std::size_t width, std::size_t height, std::size_t pixel_range = 256>
void histogram_template_test_clear()
{
	auto image = GetBag(noarr::wrap(GetImageStructreStructure<layout>::GetImageStructure()).template set_length<'x'>(width).template set_length<'y'>(height)); // image size 
	auto histogram = GetBag(noarr::array<'x', pixel_range, noarr::scalar<int>>()); // lets say that every image has 256 pixel_range

	image.clear();
	histogram.clear();

	int x_size = image.structure().template get_length<'x'>();
	int y_size = image.structure().template get_length<'y'>();

	for (int i = 0; i < x_size; i++)
	{
		for (int j = 0; j < y_size; j++)
		{
			int pixel_value = image.structure().template get_at<'x', 'y'>(image.data(), i, j);

			int& histogram_value = histogram.structure().template get_at<'x'>(histogram.data(), pixel_value);
			histogram_value = histogram_value + 1;
		}
	}
}
