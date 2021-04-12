#include <catch2/catch.hpp>

#include <cstddef>
#include <iostream>

// #include <noarr/pipelines/compute_node.hpp>
// #include <noarr/pipelines/link.hpp>
// #include <noarr/pipelines/envelope.hpp>

// class my_producer : public harbor {
// public:
//     dock output_dock;

//     bool can_advance() override {
//         return this->output_dock.has_ship && !this->output_dock.ready_to_leave;
//     }

//     // parent method
//     void scheduler_poke() {
//         if (this->can_advance())
//             return;
        
//         // TODO: asynchronous
//         this->being_advanced = true;
//         this->advance();
//         this->being_advanced = false;
//     }

//     void advance() override {
//         auto s = this->output_dock.ship;
//         s.structure = $$$;
//         s.data = $$$;

//         this->output_dock.ready_to_leave = true;
//     }
// };

#include "my_producing_harbor.hpp"

TEST_CASE("Producing harbor", "[harbor]") {
    std::cout << "Hello world!" << std::endl;

    auto prod = my_producing_harbor("lorem ipsum", 3);
    
    REQUIRE(p.foo() == 42);
}

// TEST_CASE("Two harbors can cycle a ship", "[harbor]") {
//     // std::cout << "Hello world!" << std::endl;
// }
