#include <catch2/catch.hpp>

#include <string>
#include <iostream>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>
#include <noarr/pipelines/Hub.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>

#include "raw_nodes/MyProducingNode.hpp"
#include "raw_nodes/MyConsumingNode.hpp"

using namespace noarr::pipelines;

/*
    TODO: also implement and test the in-place envelope editing
        and passing between managers:
    em.forward_returning_envelopes_to(other_em);
    em.read_write_port(Device(-1));
 */

/**
 * Tests the envelope hub in the messaging queue setup,
 * when it performs memory transfer and/or double buffering
 */
TEST_CASE("Hub as queue", "[.][hub]") { // TEST IS SKIPPED: [.]

    // create a hub (buffer count, buffer size)
    auto hub = Hub<std::size_t, char>(2, 1024);

    // create our nodes
    auto prod = MyProducingNode("lorem ipsum", 3);
    auto cons = MyConsumingNode();

    // setup a scheduler
    auto scheduler = DebuggingScheduler();
    scheduler.add(prod);
    // scheduler.add(hub); // TODO: add the hub to the scheduler
    scheduler.add(cons);
    
    SECTION("can forward traffic within one device") {
        // both sides are on the host
        Node::link_ports(prod.output_port, hub.write_port(Device(-1)));
        Node::link_ports(cons.input_port, hub.read_port(Device(-1)));

        scheduler.run();
        REQUIRE(cons.received_string == "lorem ipsum");
    }

    SECTION("can forward traffic between two devices") {
        // the consumer is on a faked GPU device
        Node::link_ports(prod.output_port, hub.write_port(Device(-1)));
        Node::link_ports(cons.input_port, hub.read_port(Device(0)));

        scheduler.run();
        REQUIRE(cons.received_string == "lorem ipsum");
    }
}
