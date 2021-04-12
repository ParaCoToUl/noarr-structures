#include <catch2/catch.hpp>

#include <cstddef>
#include <iostream>

// #include <noarr/pipelines/compute_node.hpp>
// #include <noarr/pipelines/link.hpp>
// #include <noarr/pipelines/envelope.hpp>

#include "my_producing_harbor.hpp"

TEST_CASE("Producing harbor", "[harbor]") {

    // create a host ship
    char buffer[1024];
    auto s = ship<std::size_t, char>(memory_device(-1), buffer, 1024);

    // create our producer harbor and put the ship in the dock
    auto prod = my_producing_harbor("lorem ipsum", 3);
    
    SECTION("it cannot advance without a ship") {
        REQUIRE(!prod.can_advance());
    }

    SECTION("it can advance with a ship") {
        prod.output_dock.arrive_ship(&s);
        REQUIRE(prod.can_advance());
    }

    SECTION("it can produce a chunk") {
        prod.output_dock.arrive_ship(&s);
        
        prod.scheduler_start();
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);

        REQUIRE(s.has_payload);
        REQUIRE(s.structure == 3);
        REQUIRE(s.buffer[0] == 'l');
        REQUIRE(s.buffer[1] == 'o');
        REQUIRE(s.buffer[2] == 'r');
    }

    SECTION("it can produce all chunks and stop advancing") {
        prod.output_dock.arrive_ship(&s);
        
        prod.scheduler_start();

        // chunk 0 "lor"
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(s.has_payload);
        REQUIRE(s.structure == 3);
        REQUIRE(s.buffer[0] == 'l');

        s.has_payload = false;
        prod.output_dock.ship_processed = false;

        // chunk 1 "em "
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(s.has_payload);
        REQUIRE(s.structure == 3);
        REQUIRE(s.buffer[0] == 'e');

        s.has_payload = false;
        prod.output_dock.ship_processed = false;

        // chunk 2 "ips"
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(s.has_payload);
        REQUIRE(s.structure == 3);
        REQUIRE(s.buffer[0] == 'i');

        s.has_payload = false;
        prod.output_dock.ship_processed = false;

        // chunk 3 "um"
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(s.has_payload);
        REQUIRE(s.structure == 2);
        REQUIRE(s.buffer[0] == 'u');

        s.has_payload = false;
        prod.output_dock.ship_processed = false;

        // done
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(!data_advanced);
        });
        prod.scheduler_post_update(false);

        REQUIRE(!s.has_payload);
        REQUIRE(!prod.output_dock.ship_processed);
    }
}

// TEST_CASE("Two harbors can cycle a ship", "[harbor]") {
//     // std::cout << "Hello world!" << std::endl;
// }
