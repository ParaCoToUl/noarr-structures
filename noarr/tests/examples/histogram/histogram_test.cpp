#include <catch2/catch.hpp>

#include <cstddef>
#include <iostream>
#include <array>

#include "histogram.hpp"

TEST_CASE("Histogram example", "[histogram-example]") {
    
    // create some dummy image data to test the example on
    char image_data[1024];
    image_data[0] = 42;

    // the variable that will be populated by the example
    std::array<std::size_t, 256> histogram_data;

    SECTION("it computes proper value") {
        compute_histogram((void*)image_data, histogram_data);

        // assert that the result is ok
        REQUIRE(histogram_data.at(0) == 42);
	}
}
