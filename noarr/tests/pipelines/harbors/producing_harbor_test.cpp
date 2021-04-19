#include <catch2/catch.hpp>

#include <cstddef>
#include <iostream>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>

#include "my_producing_harbor.hpp"

using namespace noarr::pipelines;

TEST_CASE("Producing harbor", "[harbor]") {

    // create an envelope
    char buffer[1024];
    auto s = Envelope<std::size_t, char>(Device(-1), buffer, 1024);

    // create our producer harbor
    auto prod = my_producing_harbor("lorem ipsum", 3);
    
    SECTION("it cannot advance without an envelope") {
        REQUIRE(!prod.can_advance());
    }

    SECTION("it can advance with an envelope") {
        prod.output_port.attach_envelope(&s);
        REQUIRE(prod.can_advance());
    }

    SECTION("it can produce a chunk") {
        prod.output_port.attach_envelope(&s);
        
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
        prod.output_port.attach_envelope(&s);
        
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
        prod.output_port.envelope_processed = false;

        // chunk 1 "em "
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(s.has_payload);
        REQUIRE(s.structure == 3);
        REQUIRE(s.buffer[0] == 'e');

        s.has_payload = false;
        prod.output_port.envelope_processed = false;

        // chunk 2 "ips"
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(s.has_payload);
        REQUIRE(s.structure == 3);
        REQUIRE(s.buffer[0] == 'i');

        s.has_payload = false;
        prod.output_port.envelope_processed = false;

        // chunk 3 "um"
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(s.has_payload);
        REQUIRE(s.structure == 2);
        REQUIRE(s.buffer[0] == 'u');

        s.has_payload = false;
        prod.output_port.envelope_processed = false;

        // done
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(!data_advanced);
        });
        prod.scheduler_post_update(false);

        REQUIRE(!s.has_payload);
        REQUIRE(!prod.output_port.envelope_processed);
    }
}
