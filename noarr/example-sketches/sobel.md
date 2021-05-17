# Sobel example sketch


**Demonstrates:**

- parallel execution on one device
- parallel read from one buffer
- double-buffering


**Description:**

The input is a sequence of N images of various resolutions with a reasonable upper limit on the resolution (to allocate buffers). For each of these images a sobel operator is calculated and saved to the output as two images (vertical edges & horizontal edges). The two edge-detection kernels run in parallel, while reading from the same source image. All images are grayscale for simplicity. This algorithm is not exactly Sobel, but it's close enough.


**Code:**

```cpp
void sobel(const std::vector<std::string>& source_images) {
    using Image = noarr::vector<'x', noarr::vector<'y'>, noarr::scalar<char>>;

    Hub<Image> image_hub;
    Hub<Image> horizontal_edges_hub;
    Hub<Image> vertical_edges_hub;

    // global state
    std::size_t image_index = 0;
    bool horizontal_done = false;
    bool vertical_done = false;

    auto loader = noarr::structures::thread_compute_node([&](builder node){
        auto& image = node.link(image_hub.write(Device::HOST_INDEX));

        node.can_advance([&](){
            return image_index < source_images.size();
        });

        node.advance([&](){
            node.run_in_other_thread([&](){
                // pseudo:
                // image.envelope.structure =
                // image.envelope.buffer =      LOAD_IMAGE(source_images[image_index])

                image.envelope.structure = Image()
                    | noarr::resize<'x'>(width)
                    | noarr::resize<'y'>(height);
            });
        });

        node.finalize([&](){
            ++image_index;
        });
    });

    // helper method called when either horizontal or vertical kernel finishes
    auto check_both_edges_finished = [&](){
        if (horizontal_done && vertical_done) {
            image_hub.consume_latest_chunk();
            horizontal_done = false;
            vertical_done = false;
        }
    };

    auto horizontal_filter = noarr::structures::cuda_compute_node([&](builder node){
        auto& image = node.link(image_hub.readpeek(Device::DEVICE_INDEX));
        auto& horizontal_edges = node.link(horizontal_edges_hub.write(Device::DEVICE_INDEX));

        node.can_advance([&](){
            return true;
        });

        node.advance([&](){
            // pseudo: run_horizontal_filter_kernel(image, horizontal_edges)
            
            cudaSynchronize(node.cuda_stream, node.calback);
        });

        node.finalize([&](){
            horizontal_done = true;
            check_both_edges_done();
        });
    });

    auto vertical_filter = ...; // same, only replace horizontal with vertical

    auto saver = noarr::structures::thread_compute_node([&](builder node){
        auto& horizontal_edges = node.link(horizontal_edges_hub.read(Device::HOST_INDEX));
        auto& vertical_edges = node.link(vertical_edges_hub.read(Device::HOST_INDEX));

        node.can_advance([&](){
            return true;
        });

        node.advance([&](){
            node.run_in_other_thread([&](){
                // pseudo:
                // save two files - one for horizontal and the other for vertical
                // NOTE: this node can be split to two, that could be run in parallel
                // but if both directions were in one file it would have to be this way
            });
        });
    });

    // setup dataflow strategy
    image_hub.set_dataflow_strategy.to_links(horizontal_filter, vertical_filter);
    horizontal_edges_hub.set_dataflow_strategy.to_link(saver);
    vertical_edges_hub.set_dataflow_strategy.to_link(saver);

    // setup scheduler
    auto sched = noarr::pipelines::scheduler();
    sched.add(image_hub);
    sched.add(horizontal_edges_hub);
    sched.add(vertical_edges_hub);
    sched.add(loader);
    sched.add(horizontal_filter);
    sched.add(vertical_filter);
    sched.add(saver);

    // run
    sched.run();
}
```
