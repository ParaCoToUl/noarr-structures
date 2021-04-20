#include <catch2/catch.hpp>

#include <iostream>
#include <string>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>

#include "MyConsumingNode.hpp"

using namespace noarr::pipelines;

TEST_CASE("Consuming node", "[node]") {

    // create an envelope
    char buffer[1024];
    auto env = Envelope<std::size_t, char>(Device(-1), buffer, 1024);

    // create our consumer node
    auto cons = MyConsumingNode();

    // setup a scheduler
    auto scheduler = DebuggingScheduler();
    scheduler.add(cons);

    SECTION("it cannot advance without an envelope") {
        REQUIRE(!cons.can_advance());
    }

    SECTION("it can advance with an envelope") {
        cons.input_port.attach_envelope(env);
        REQUIRE(cons.can_advance());
    }

    SECTION("it can consume a chunk") {
        env.has_payload = true;
        env.structure = 3;
        env.buffer[0] = 'l';
        env.buffer[1] = 'o';
        env.buffer[2] = 'r';
        
        cons.input_port.attach_envelope(env);
        
        REQUIRE(scheduler.update_next_node());
        REQUIRE(!env.has_payload);
        REQUIRE(cons.received_string == "lor");
    }
}
