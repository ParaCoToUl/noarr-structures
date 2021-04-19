#include <catch2/catch.hpp>

#include <cstddef>
#include <iostream>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>

#include "MyProducingNode.hpp"

using namespace noarr::pipelines;

TEST_CASE("Producing node", "[node]") {

    // create an envelope
    char buffer[1024];
    auto env = Envelope<std::size_t, char>(Device(-1), buffer, 1024);

    // create our producer node
    auto prod = MyProducingNode("lorem ipsum", 3);
    
    SECTION("it cannot advance without an envelope") {
        REQUIRE(!prod.can_advance());
    }

    SECTION("it can advance with an envelope") {
        prod.output_port.attach_envelope(&env);
        REQUIRE(prod.can_advance());
    }

    SECTION("it can produce a chunk") {
        prod.output_port.attach_envelope(&env);
        
        prod.scheduler_start();
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);

        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 3);
        REQUIRE(env.buffer[0] == 'l');
        REQUIRE(env.buffer[1] == 'o');
        REQUIRE(env.buffer[2] == 'r');
    }

    SECTION("it can produce all chunks and stop advancing") {
        prod.output_port.attach_envelope(&env);
        
        prod.scheduler_start();

        // chunk 0 "lor"
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 3);
        REQUIRE(env.buffer[0] == 'l');

        env.has_payload = false;
        prod.output_port.envelope_processed = false;

        // chunk 1 "em "
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 3);
        REQUIRE(env.buffer[0] == 'e');

        env.has_payload = false;
        prod.output_port.envelope_processed = false;

        // chunk 2 "ips"
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 3);
        REQUIRE(env.buffer[0] == 'i');

        env.has_payload = false;
        prod.output_port.envelope_processed = false;

        // chunk 3 "um"
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(data_advanced);
        });
        prod.scheduler_post_update(true);
        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 2);
        REQUIRE(env.buffer[0] == 'u');

        env.has_payload = false;
        prod.output_port.envelope_processed = false;

        // done
        prod.scheduler_update([](bool data_advanced){
            REQUIRE(!data_advanced);
        });
        prod.scheduler_post_update(false);

        REQUIRE(!env.has_payload);
        REQUIRE(!prod.output_port.envelope_processed);
    }
}
