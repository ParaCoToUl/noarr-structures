#include <catch2/catch.hpp>

#include <cstddef>
#include <string>
#include <iostream>

#include <noarr/pipelines/memory_device.hpp>
#include <noarr/pipelines/ship.hpp>
#include <noarr/pipelines/scheduler.hpp>

#include "harbors/my_producing_harbor.hpp"
#include "harbors/my_consuming_harbor.hpp"

/*
    This is a playground for developing a lambda-based API for pipeline building
 */

using namespace noarr::pipelines;

/*
TEST_CASE("Lambda pipeline", "[pipeline]") {
    REQUIRE(true);

    std::string message_to_send = "Lorem ipsum dolor sit amet.";
    std::size_t at_index = 0;
    std::size_t chunk_size = 3;

    std::string message_received = "";

    auto env = envelope<std::size_t, char>(
        memory_device(-1), // from
        memory_device(-1) // to
    );

    auto prod = compute_node::create()
        .link(env.write_dock)
        .advance([&](
            std::function<void()> callback,
            ship<std::size_t, char>& ship
        ){
            std::size_t items_to_take = std::min(
                chunk_size,
                message_to_send.length() - at_index
            );

            message_to_send.copy(ship.buffer, items_to_take, at_index);
            ship.structure = items_to_take;

            at_index += items_to_take;

            callback();
        });

    auto cons = compute_node::create()
        .link(env.read_dock)
        .advance([&](
            std::function<void()> callback,
            ship<std::size_t, char>& ship
        ){
            message_received.append(ship.buffer, ship.structure);

            callback();
        });

    auto sched = scheduler();
    sched.add(env);
    sched.add(prod);
    sched.add(cons);

    sched.run();

    REQUIRE(message_received == message_to_send);
}
*/
