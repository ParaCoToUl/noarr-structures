#include <catch2/catch.hpp>

#include <string>
#include <memory>
#include <iostream>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Buffer.hpp>

using namespace noarr::pipelines;

TEST_CASE("Buffer", "[pipelines][unit][buffer]") {
    
    SECTION("can be created from existing pointer") {
        void* ptr = malloc(1024);
        
        Buffer b = Buffer::from_existing(Device::HOST_INDEX, ptr, 1024);
        
        REQUIRE(b.data_pointer == ptr);
        REQUIRE(b.bytes == 1024);
        REQUIRE(b.device_index == Device::HOST_INDEX);
        REQUIRE(b.allocator == nullptr);

        free(ptr);
    };

    SECTION("can be moved") {
        void* ptr = malloc(1024);
        
        Buffer b = Buffer::from_existing(Device::HOST_INDEX, ptr, 1024);

        Buffer c = std::move(b);

        REQUIRE(b.data_pointer == nullptr);
        REQUIRE(b.allocator == nullptr);
        
        REQUIRE(c.data_pointer == ptr);
        REQUIRE(c.allocator == nullptr);

        free(ptr);
    };
}
