#include <catch2/catch.hpp>

#include <cstddef>
#include <iostream>

#include <noarr/pipelines/memory_device.hpp>
#include <noarr/pipelines/ship.hpp>

#include "my_consuming_harbor.hpp"

using namespace noarr::pipelines;

TEST_CASE("Consuming harbor", "[harbor]") {

    // create a ship
    char buffer[1024];
    auto s = ship<std::size_t, char>(memory_device(-1), buffer, 1024);

    // create our consumer harbor
    auto cons = my_consuming_harbor();

    SECTION("it cannot advance without a ship") {
        REQUIRE(!cons.can_advance());
    }

    SECTION("it can advance with a ship") {
        cons.input_dock.arrive_ship(&s);
        REQUIRE(cons.can_advance());
    }

    SECTION("it can consume a chunk") {
        s.has_payload = true;
        s.structure = 3;
        s.buffer[0] = 'l';
        s.buffer[1] = 'o';
        s.buffer[2] = 'r';
        
        cons.input_dock.arrive_ship(&s);
        
        cons.scheduler_start();
        cons.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        cons.scheduler_post_update(true);

        REQUIRE(!s.has_payload);
        REQUIRE(cons.received_string == "lor");
    }
}
