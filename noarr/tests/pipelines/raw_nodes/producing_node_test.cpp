#include <catch2/catch.hpp>

#include <iostream>
#include <string>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>

#include "MyProducingNode.hpp"

using namespace noarr::pipelines;

TEST_CASE("Producing node", "[node]") {

    // create an envelope
    char buffer[1024];
    auto env = Envelope<std::size_t, char>(Device(-1), buffer, 1024);

    // create our producer node
    auto prod = MyProducingNode("lorem ipsum", 3);

    // setup a scheduler
    auto scheduler = DebuggingScheduler();
    scheduler.add(prod);
    
    SECTION("it cannot advance without an envelope") {
        REQUIRE(!prod.can_advance());
    }

    SECTION("it can advance with an envelope") {
        prod.output_port.attach_envelope(env);
        REQUIRE(prod.can_advance());
    }

    SECTION("it can produce a chunk") {
        prod.output_port.attach_envelope(env);
        
        REQUIRE(scheduler.update_next_node());

        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 3);
        REQUIRE(std::string(env.buffer, 3) == "lor");
    }

    SECTION("it can produce all chunks and stop advancing") {
        prod.output_port.attach_envelope(env);
        
        // chunk 0 "lor"
        REQUIRE(scheduler.update_next_node());
        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 3);
        REQUIRE(std::string(env.buffer, 3) == "lor");

        env.has_payload = false;
        prod.output_port.set_processed(false);

        // chunk 1 "em "
        REQUIRE(scheduler.update_next_node());
        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 3);
        REQUIRE(std::string(env.buffer, 3) == "em ");

        env.has_payload = false;
        prod.output_port.set_processed(false);

        // chunk 2 "ips"
        REQUIRE(scheduler.update_next_node());
        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 3);
        REQUIRE(std::string(env.buffer, 3) == "ips");

        env.has_payload = false;
        prod.output_port.set_processed(false);

        // chunk 3 "um"
        REQUIRE(scheduler.update_next_node());
        REQUIRE(env.has_payload);
        REQUIRE(env.structure == 2);
        REQUIRE(std::string(env.buffer, 2) == "um");

        env.has_payload = false;
        prod.output_port.set_processed(false);

        // done
        REQUIRE(!scheduler.update_next_node());
        REQUIRE(!env.has_payload);
        REQUIRE(prod.output_port.state() != PortState::processed);
    }
}
