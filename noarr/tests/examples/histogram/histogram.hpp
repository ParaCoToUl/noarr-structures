#include <cstddef>
#include <iostream>

#include <noarr/structures/structs.hpp>
#include <noarr/structures/funcs.hpp>
#include <noarr/structures/io.hpp>
#include <noarr/structures/struct_traits.hpp>

void compute_histogram(
    void* image_data,
    std::array<std::size_t, 256>& histogram_data
) {
    // treat the image_data as raw bytes of some image format,
    // say first 8 bytes tell the width and height, and then the image data comes
    // (or something else, figure out what's best)

    histogram_data[0] = ((char*)image_data)[0];
}
