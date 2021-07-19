#include <catch2/catch.hpp>

#include <string>
#include <iostream>
#include <vector>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>
#include <noarr/pipelines/Hub.hpp>
#include <noarr/pipelines/HardwareManager.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>
#include <noarr/pipelines/LambdaComputeNode.hpp>

using namespace noarr::pipelines;

TEST_CASE("Producer filter consumer example", "[pipelines][integration][prod_fil_con]") {
    HardwareManager::default_manager().register_dummy_gpu();

    // [producer] --> [hub A] --> [filter] --> [hub B] --> [consumer]
    
    std::vector<std::int32_t> input = {1, 2, 3, 4, 5};
    std::vector<std::int32_t> output = {};
    std::vector<std::int32_t> expected_output = {1, 3, 5};
    std::size_t sent_items = 0;
    
    auto hub_A = Hub<std::size_t, std::int32_t>(sizeof(std::int32_t) * 1);
    hub_A.allocate_envelopes(Device::HOST_INDEX, 2);
    hub_A.allocate_envelopes(Device::DUMMY_GPU_INDEX, 2);
    hub_A.set_max_queue_length(2);

    auto hub_B = Hub<std::size_t, std::int32_t>(sizeof(std::int32_t) * 1);
    hub_B.allocate_envelopes(Device::HOST_INDEX, 2);
    hub_B.allocate_envelopes(Device::DUMMY_GPU_INDEX, 2);
    hub_B.set_max_queue_length(2);

    auto producer = LambdaComputeNode("producer");
    auto filter = LambdaComputeNode("filter");
    auto consumer = LambdaComputeNode("consumer");

    /* producer */ {
        auto& link = producer.link(hub_A.to_produce(Device::HOST_INDEX));

        producer.can_advance([&](){
            return sent_items < input.size();
        });

        producer.advance([&](){
            link.envelope->buffer[0] = input[sent_items];
            producer.callback();
        });

        producer.post_advance([&](){
            sent_items += 1;
        });
    }

    /* filter */ {
        auto& link_A = filter.link(hub_A.to_consume(Device::DUMMY_GPU_INDEX));
        auto& link_B = filter.link(hub_B.to_produce(Device::DUMMY_GPU_INDEX, false));

        filter.advance([&](){
            std::int32_t item = link_A.envelope->buffer[0];
            
            // pass on only odd elements
            if (item % 2 == 1) {
                link_B.envelope->buffer[0] = item;
                link_B.commit();
            }
            
            filter.callback();
        });
    }

    /* consumer */ {
        auto& link = consumer.link(hub_B.to_consume(Device::HOST_INDEX));

        consumer.advance([&](){
            output.push_back(link.envelope->buffer[0]);
            consumer.callback();
        });
    }

    auto scheduler = DebuggingScheduler();
    scheduler.add(hub_A);
    scheduler.add(hub_B);
    scheduler.add(producer);
    scheduler.add(filter);
    scheduler.add(consumer);

    scheduler.run();

    REQUIRE(output.size() == expected_output.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        REQUIRE(expected_output[i] == output[i]);
    }
}
