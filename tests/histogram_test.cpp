#include <catch2/catch_test_macros.hpp>

#include <array>
#include <iostream>
#include <tuple>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/bag.hpp>


enum class ImageDataLayout { ArrayOfArrays = 1, VectorOfVectors = 2 };

template<ImageDataLayout layout>
struct GetImageStructureGetter;

template<>
struct GetImageStructureGetter<ImageDataLayout::ArrayOfArrays>
{
	static constexpr auto GetImageStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<>
struct GetImageStructureGetter<ImageDataLayout::VectorOfVectors>
{
	static constexpr auto GetImageStructure()
	{
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<ImageDataLayout layout, std::size_t width, std::size_t height, std::size_t pixel_range = 256>
void histogram_template_test()
{
	auto image = noarr::make_bag(noarr::wrap(GetImageStructureGetter<layout>::GetImageStructure()).template set_length<'x'>(width).template set_length<'y'>(height));

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

			REQUIRE(pixel_value == 0);

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
