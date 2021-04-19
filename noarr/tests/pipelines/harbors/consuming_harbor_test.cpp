#include <catch2/catch.hpp>

#include <cstddef>
#include <iostream>

#include <noarr/pipelines/memory_device.hpp>
#include <noarr/pipelines/Envelope.hpp>

#include "my_consuming_harbor.hpp"

using namespace noarr::pipelines;

TEST_CASE("Consuming harbor", "[harbor]") {

    // create an envelope
    char buffer[1024];
    auto env = Envelope<std::size_t, char>(memory_device(-1), buffer, 1024);

    // create our consumer harbor
    auto cons = my_consuming_harbor();

    SECTION("it cannot advance without a ship") {
        REQUIRE(!cons.can_advance());
    }

    SECTION("it can advance with a ship") {
        cons.input_dock.attach_envelope(&env);
        REQUIRE(cons.can_advance());
    }

    SECTION("it can consume a chunk") {
        env.has_payload = true;
        env.structure = 3;
        env.buffer[0] = 'l';
        env.buffer[1] = 'o';
        env.buffer[2] = 'r';
        
        cons.input_dock.attach_envelope(&env);
        
        cons.scheduler_start();
        cons.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        cons.scheduler_post_update(true);

        REQUIRE(!env.has_payload);
        REQUIRE(cons.received_string == "lor");
    }
}
