#include <catch2/catch.hpp>

#include <array>
#include <iostream>
#include <tuple>

#include "noarr/structures_extended.hpp"

TEST_CASE("Image", "[image]") {
	//noarr::array<'x', 1920, noarr::array<'y', 1080, noarr::tuple<'p', noarr::scalar<float>, noarr::scalar<float>, noarr::scalar<float>, noarr::scalar<float>>>> image;

	noarr::array<'x', 1920, noarr::array<'y', 1080, noarr::array<'p', 4, noarr::scalar<float>>>> g;
	auto grayscale = noarr::wrap(g);

	std::vector<char> my_blob_buffer(grayscale.get_size());
	char* my_blob = (char*)my_blob_buffer.data();

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(grayscale)>::value);
		//REQUIRE(!noarr::is_cube<decltype(image)>::value);
	}

	
	SECTION("check TODO") {
		//auto value_ref = image | fix<'x'>(0) | fix<'y'>(0) | fix<'p'>(2);
		//std::declval

		//REQUIRE((typeid(image | noarr::fix<'x'>(0) | noarr::fix<'y'>(0) | noarr::fix<'p'>(2)).name()) == "float");
		//auto value_ref = image | noarr::fix<'x'>(0) | noarr::fix<'y'>(0) | noarr::fix<'p'>(2);
		//std::size_t offset = image | noarr::fix<'x', 'y', 'p'>(0, 0, 2) | noarr::offset();
		std::size_t offset = grayscale.offset<'x', 'y', 'p'>(0, 0, 2);
		float& value_ref = *((float*)(my_blob + offset));
		value_ref = 0;
		//float& value_ref = image | fix<'x'>(0) | fix<'y'>(0) | fix<'p'>(2) | offset();
	}
}

TEST_CASE("Pipes Vector", "[resizing]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto v2 = v | noarr::set_length<'x'>(10); // transform

	SECTION("size check 1") {
		REQUIRE((v2 | noarr::get_length<'x'>()) == 10);
	}

	auto v3 = v  | noarr::set_length<'x'>(20); // transform
	auto v4 = v2 | noarr::set_length<'x'>(30); // transform

	SECTION("size check 2") {
		REQUIRE((v2 | noarr::get_length<'x'>()) == 10);
		REQUIRE((v3 | noarr::get_length<'x'>()) == 20);
		REQUIRE((v4 | noarr::get_length<'x'>()) == 30);
	}

	SECTION("check is_cube 2") {
		REQUIRE(!noarr::is_cube<decltype(v)>::value);
		REQUIRE( noarr::is_cube<decltype(v2)>::value);
		REQUIRE( noarr::is_cube<decltype(v3)>::value);
		REQUIRE( noarr::is_cube<decltype(v4)>::value);
	}

	auto v5 = v4 | noarr::set_length<'x'>(-10); // transform

	SECTION("size check 3") {
		// REQUIRE((v5 | noarr::get_size()) == -10); FIXME: this is absolutely crazy
	}

	SECTION("check is_cube 3") {
		REQUIRE(noarr::is_cube<decltype(v5)>::value);
	}
}



TEST_CASE("Pipes Vector2", "[is_trivial]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto v2 = v | noarr::set_length<'x'>(10); // transform

	SECTION("is_trivial check 1") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);
	}

	auto v3 = v | noarr::set_length<'x'>(20); // transform
	auto v4 = v2 | noarr::set_length<'x'>(30); // transform

	SECTION("is_trivial check 2") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);

		REQUIRE(std::is_trivial<decltype(v3)>::value);
		REQUIRE(std::is_standard_layout<decltype(v3)>::value);

		REQUIRE(std::is_trivial<decltype(v4)>::value);
		REQUIRE(std::is_standard_layout<decltype(v4)>::value);
	}
}


TEST_CASE("Pipes Array", "[is_trivial]")
{
	noarr::array<'x', 1920, noarr::scalar<float>> v;
	auto v2 = v | noarr::set_length<'x'>(10); // transform

	SECTION("is_trivial check 1") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);
	}

	auto v3 = v | noarr::set_length<'x'>(20); // transform
	auto v4 = v2 | noarr::set_length<'x'>(30); // transform

	SECTION("is_trivial check 2") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);

		REQUIRE(std::is_trivial<decltype(v3)>::value);
		REQUIRE(std::is_standard_layout<decltype(v3)>::value);

		REQUIRE(std::is_trivial<decltype(v4)>::value);
		REQUIRE(std::is_standard_layout<decltype(v4)>::value);
	}
}




//////////
// Dots //
//////////




TEST_CASE("Vector", "[resizing]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto w = noarr::wrap(v);
	auto v2 = w.set_length<'x'>(10); // transform

	SECTION("size check 1") {
		REQUIRE((v2.get_length<'x'>()) == 10);
	}

	auto v3 = w.set_length<'x'>(20); // transform
	auto v4 = v2.set_length<'x'>(30); // transform

	SECTION("size check 2") {
		REQUIRE((v2.get_length<'x'>()) == 10);
		REQUIRE((v3.get_length<'x'>()) == 20);
		REQUIRE((v4.get_length<'x'>()) == 30);
	}

	SECTION("check is_cube 2") {
		REQUIRE(!noarr::is_cube<decltype(v)>::value);
		REQUIRE(noarr::is_cube<decltype(v2)>::value);
		REQUIRE(noarr::is_cube<decltype(v3)>::value);
		REQUIRE(noarr::is_cube<decltype(v4)>::value);
	}

	auto v5 = v4.set_length<'x'>(-10); // transform

	SECTION("size check 3") {
		// REQUIRE((v5 | noarr::get_size()) == -10); FIXME: this is absolutely crazy
	}

	SECTION("check is_cube 3") {
		REQUIRE(noarr::is_cube<decltype(v5)>::value);
	}
}



TEST_CASE("Vector2", "[is_trivial]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto w = noarr::wrap(v);
	auto v2 = w.set_length<'x'>(10); // transform

	SECTION("is_trivial check 1") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);
	}

	auto v3 = w.set_length<'x'>(20); // transform
	auto v4 = v2.set_length<'x'>(30); // transform

	SECTION("is_trivial check 2") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);

		REQUIRE(std::is_trivial<decltype(v3)>::value);
		REQUIRE(std::is_standard_layout<decltype(v3)>::value);

		REQUIRE(std::is_trivial<decltype(v4)>::value);
		REQUIRE(std::is_standard_layout<decltype(v4)>::value);
	}
}


TEST_CASE("Array", "[is_trivial]")
{
	noarr::array<'x', 1920, noarr::scalar<float>> v;
	auto w = noarr::wrap(v);
	auto v2 = w.set_length<'x'>(10); // transform

	SECTION("is_trivial check 1") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);
	}

	auto v3 = w.set_length<'x'>(20); // transform
	auto v4 = v2.set_length<'x'>(30); // transform

	SECTION("is_trivial check 2") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);

		REQUIRE(std::is_trivial<decltype(v3)>::value);
		REQUIRE(std::is_standard_layout<decltype(v3)>::value);

		REQUIRE(std::is_trivial<decltype(v4)>::value);
		REQUIRE(std::is_standard_layout<decltype(v4)>::value);
	}
}

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
			int pixel_value = image.template at<'x','y'>(i, j); // v3

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
