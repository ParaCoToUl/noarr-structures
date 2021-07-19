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

TEST_CASE("Producer modifier consumer example", "[pipelines][integration][prod_mod_con]") {
    HardwareManager::default_manager().register_dummy_gpu();
    
    std::vector<std::int32_t> input = {1, 2, 3, 4, 5};
    std::vector<std::int32_t> output = {};
    std::vector<std::int32_t> expected_output = {1, 4, 9, 16, 25};
    std::size_t sent_items = 0;
    
    auto item_hub = Hub<std::size_t, std::int32_t>(sizeof(std::int32_t) * 1);
    item_hub.allocate_envelopes(Device::HOST_INDEX, 2);
    item_hub.allocate_envelopes(Device::DUMMY_GPU_INDEX, 2);
    item_hub.set_max_queue_length(1);

    auto producer = LambdaComputeNode("producer");
    auto modifier = LambdaComputeNode("modifier");
    auto consumer = LambdaComputeNode("consumer");

    /* producer */ {
        auto& item_link = producer.link(item_hub.to_produce(Device::HOST_INDEX));

        producer.can_advance([&](){
            return sent_items < input.size();
        });

        producer.advance([&](){
            item_link.envelope->buffer[0] = input[sent_items];
            producer.callback();
        });

        producer.post_advance([&](){
            sent_items += 1;
        });
    }

    /* modifier */ {
        auto& item_link = modifier.link(item_hub.to_modify(Device::DUMMY_GPU_INDEX));

        modifier.advance([&](){
            std::int32_t item = item_link.envelope->buffer[0];
            item = item * item;
            item_link.envelope->buffer[0] = item;
            modifier.callback();
        });

        modifier.post_advance([&](){
            item_hub.flow_data_to(consumer);
        });
    }

    /* consumer */ {
        auto& item_link = consumer.link(item_hub.to_consume(Device::HOST_INDEX));

        consumer.advance([&](){
            output.push_back(item_link.envelope->buffer[0]);
            consumer.callback();
        });

        consumer.post_advance([&](){
            item_hub.flow_data_to(modifier);
        });
    }

    item_hub.flow_data_to(modifier);

    auto scheduler = DebuggingScheduler();
    scheduler.add(item_hub);
    scheduler.add(producer);
    scheduler.add(modifier);
    scheduler.add(consumer);

    scheduler.run();

    REQUIRE(output.size() == expected_output.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        REQUIRE(expected_output[i] == output[i]);
    }
}
