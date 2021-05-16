// #include <catch2/catch.hpp>

// #include <string>
// #include <iostream>

// #include <noarr/pipelines/Device.hpp>
// #include <noarr/pipelines/Envelope.hpp>
// #include <noarr/pipelines/Hub.hpp>
// #include <noarr/pipelines/DebuggingScheduler.hpp>

// #include "raw_nodes/MyProducingNode.hpp"
// #include "raw_nodes/MyConsumingNode.hpp"

// using namespace noarr::pipelines;

// /**
//  * Tests the envelope hub in the messaging queue setup,
//  * when it performs memory transfer and/or double buffering
//  */
// TEST_CASE("Hub as queue", "[hub]") {

//     // create a hub
//     auto hub = Hub<std::size_t, char>();

//     // create our nodes
//     auto prod = MyProducingNode("lorem ipsum", 3);
//     auto cons = MyConsumingNode();

//     // setup a scheduler
//     auto scheduler = DebuggingScheduler(std::cout);
//     scheduler.add(prod);
//     scheduler.add(hub);
//     scheduler.add(cons);
    
//     SECTION("can forward traffic within one device") {
//         // both sides are on the host
//         hub.write(Device::HOST_INDEX).attach_to_port(prod.output_port);
//         hub.read(Device::HOST_INDEX).attach_to_port(cons.input_port);

//         REQUIRE(true);

//         // scheduler.run();
//         // REQUIRE(cons.received_string == "lorem ipsum");
//     }

//     // TODO: implement device copying logic
//     // SECTION("can forward traffic between two devices") {
//     //     // the consumer is on a faked GPU device
//     //     hub.write(Device::HOST_INDEX).attach_to_port(prod.output_port);
//     //     hub.read(Device::DEVICE_INDEX).attach_to_port(cons.input_port);

//     //     scheduler.run();
//     //     REQUIRE(cons.received_string == "lorem ipsum");
//     // }
// }
