# Autobalance example sketch


**Demonstrates:**

- modifying envelopes in-place and passing them to a different hub
    (= in-place double-buffering)
    (achieved by swapping envelope contents)


**Description:**

The input is a sequence of N images of various resolutions with a reasonable upper limit on the resolution (to allocate buffers). Each of these images is send to the GPU, maximum and minimum brightness is computed and then the image is scaled in-place to use the full range of 0-255 brightness levels. The image is then sent back to the CPU. All images are grayscale for simplicity.


**Code:**

```cpp
void sobel(const std::vector<std::string>& source_images) {
    using Image = noarr::vector<'x', noarr::vector<'y'>, noarr::scalar<char>>;

    Hub<Image> raw_image_hub;
    Hub<Image> adjusted_image_hub;

    // global state
    std::size_t image_index = 0;

    auto loader = noarr::structures::thread_compute_node([&](builder node){
        auto& raw_image = node.link(image_hub.write(Device::HOST_INDEX));

        node.can_advance([&](){
            return image_index < source_images.size();
        });

        node.advance([&](){
            node.run_in_other_thread([&](){
                // pseudo:
                // raw_image.envelope.structure =
                // raw_image.envelope.buffer =      LOAD_IMAGE(source_images[image_index])

                raw_image.envelope.structure = Image()
                    | noarr::resize<'x'>(width)
                    | noarr::resize<'y'>(height);
            });
        });

        node.finalize([&](){
            ++image_index;
        });
    });

    auto adjuster = noarr::structures::cuda_compute_node([&](builder node){
        auto& raw_image = node.link(image_hub.read(Device::DEVICE_INDEX));
        auto& adjusted_image = node.link(adjusted_image_hub.write(Device::DEVICE_INDEX));

        node.can_advance([&](){
            return true;
        });

        node.advance([&](){
            // this line also checks that sizes, types and devices match
            raw_image.envelope.swap_contents_with(adjusted_image.envelope);
            
            // pseudo:
            // char min, max = run_reduction_kernel(adjusted_image.envelope)
            // run_adjusting_kernel(adjusted_image.envelope, min, max)
            
            cudaSynchronize(node.cuda_stream, node.calback);
        });
    });

    auto saver = noarr::structures::thread_compute_node([&](builder node){
        auto& adjusted_image = node.link(adjusted_image_hub.read(Device::HOST_INDEX));

        node.can_advance([&](){
            return true;
        });

        node.advance([&](){
            node.run_in_other_thread([&](){
                // pseudo:
                // save the value of adjusted_image.envelope to a file
            });
        });
    });

    // setup dataflow strategy
    raw_image_hub.set_dataflow_strategy.to_link(adjuster);
    adjusted_image_hub.set_dataflow_strategy.to_link(saver);

    // setup scheduler
    auto sched = noarr::pipelines::scheduler();
    sched.add(raw_image_hub);
    sched.add(adjusted_image_hub);
    sched.add(loader);
    sched.add(adjuster);
    sched.add(saver);

    // run
    sched.run();
}
```
