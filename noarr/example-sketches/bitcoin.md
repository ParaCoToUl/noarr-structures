# Bitcoin example sketch


**Demonstrates:**

- parallel execution on multiple devices
- parallel read from one buffer, distributed to all devices
- conditional write to a hub


**Description:**

The input is a binary message. It is loaded into memory on each device and kernels are executed (one for each device with maximum possible thread count). Each thread tries to generate a random salt, that if appended to the message and hashed would produce a hash with N leading zeros (binary). Basically this is a naive version of the proof of work algorithm of bitcoin.


**Code:**

```cpp
// returns true if the salt was found in the given number of tries
bool bitcoin(
    // input:
    const std::string& given_message,
    std::size_t max_tries,
    std::size_t salt_characters,
    std::size_t target_zero_bits,
    
    // output:
    std::string& found_salt
) {
    Hub<std::size_t, char> message_hub;
    Hub<std::size_t, char> salt_hub;

    // global state
    bool initialized = false;
    bool salt_found = false;
    std::size_t performed_tries = 0;

    auto initializer = noarr::structures::compute_node([&](builder node){
        auto message = node.link(message_hub.write(Device::HOST_INDEX));

        node.can_advance([&](){
            return !initialized;
        });

        node.advance([&](){
            points.envelope.structure = given_message.size();
            for (std::size_t i = 0; i < given_message.size(); ++i)
                points.envelope.buffer[i] = given_message[i];

            initialized = true;
            node.callback();
        });
    });

    // for each available device, create a miner
    auto miners = std::vector<CudaComputeNode>();
    for (std::size_t device_index = 0; device_index < DEVICE_COUNT; ++device_index) {
        
        // maybe do something better for heterogenous devices...
        std::size_t THREADS_PER_DEVICE = 1024;

        auto miner = noarr::structures::cuda_compute_node([&](builder node){
            auto message = node.link(message_hub.readpeek(device_index));
            auto salt = node.link(salt_hub.conditionalwrite(device_index));

            node.can_advance([&](){
                return initialized && !salt_found && performed_tries < max_tries;
            });

            node.advance([&](){
                // pseudo:
                // __device__ bool salt_found = false;
                // mine_bitcoin_kernel<<<THREAD_PER_DEVICE>>>(
                //  message, salt, salt_characters, target_zero_bits, salt_found
                // )

                if (salt_found) {
                    salt.write_did_happen(); // chunk of data was produced
                }

                cudaSynchronize(node.cuda_stream, node.calback);
            });

            node.finalize([&](){
                performed_tries += THREADS_PER_DEVICE;
            });
        });

        miners.push_back(std::move(miner));

    }

    auto finalizer = noarr::structures::compute_node([&](builder node){
        auto salt = node.link(salt_hub.read(Device::HOST_INDEX));

        node.can_advance([&](){
            return true; // conditioned by having a chunk of data to process
        });

        node.advance([&](){
            // pseudo: write_envelope_values_to_output_arguments()
            // for (char c in salt.envelope.buffer)
            //     computed_salt.push_back(c)

            node.callback();
        });
    });

    // setup dataflow strategy
    message_hub.set_dataflow_strategy.to_links(miners);
    salt_hub.set_dataflow_strategy.to_link(finalizer);

    // setup scheduler
    auto sched = noarr::pipelines::scheduler();
    sched.add(message_hub);
    sched.add(salt_hub);
    sched.add(initializer);
    sched.add_many(miners);
    sched.add(finalizer);

    // run
    sched.run();

    return salt_found;
}
```
