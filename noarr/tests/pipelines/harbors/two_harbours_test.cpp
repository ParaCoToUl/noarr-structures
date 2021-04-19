#include <catch2/catch.hpp>

#include <cstddef>
#include <iostream>

#include <noarr/pipelines/memory_device.hpp>
#include <noarr/pipelines/Envelope.hpp>
#include <noarr/pipelines/scheduler.hpp>

#include "my_producing_harbor.hpp"
#include "my_consuming_harbor.hpp"

using namespace noarr::pipelines;

TEST_CASE("Two harbors", "[harbor]") {

    // create an envelope
    char buffer[1024];
    auto s = Envelope<std::size_t, char>(memory_device(-1), buffer, 1024);

    // create our harbors
    auto prod = my_producing_harbor("lorem ipsum", 3);
    auto cons = my_consuming_harbor();

    // link those harbors together
    prod.output_dock.send_processed_envelopes_to(&cons.input_dock);
    cons.input_dock.send_processed_envelopes_to(&prod.output_dock);

    // put the ship into the producer
    prod.output_dock.attach_envelope(&s);

    // setup a scheduler
    auto sched = scheduler();
    sched.add(&prod);
    sched.add(&cons);
    
    SECTION("they can cycle a ship") {
        // run the pipeline to completion
        sched.run();
        
        // assert the string has been transfered
        REQUIRE(cons.received_string == "lorem ipsum");
    }
}
