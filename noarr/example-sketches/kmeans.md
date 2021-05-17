# k-means example sketch


**Demonstrates:**

- changing hub dataflow during execution
- in-place modification of a buffer inside a hub


**Description:**

Given a set of 2D points and a number k, find k centroids that cluster the points well. The algorithm starts by randomly choosing centroids. Then two phases repeat 1) cluster assignment 2) centroid update. The number of refinements made is also a parameter.


**Code:**

```cpp
struct point_t {
    float x, y;
}

void kmeans(
    // input:
    const std::vector<point_t>& given_points,
    std::size_t k,
    std::size_t refinements,

    // output:
    std::vector<point_t>& computed_centroids
) {
    Hub<std::size_t, point_t> points_hub;
    Hub<std::size_t, std::uint8> assignments_hub;
    Hub<std::size_t, point_t> centroids_hub;

    // global state
    bool initialized = false;
    std::size_t next_iteration = 0;

    auto initializer = noarr::structures::compute_node([&](builder node){
        auto points = node.link(points_hub.write(Device::HOST_INDEX));
        auto assignments = node.link(assignments_hub.write(Device::HOST_INDEX));
        auto centroids = node.link(centroids_hub.write(Device::HOST_INDEX));

        node.can_advance([&](){
            return !initialized;
        });

        node.advance([&](){
            points.envelope.structure = given_points.size();
            for (std::size_t i = 0; i < given_points.size(); ++i)
                points.envelope.buffer[i] = given_points[i];

            // pseudo: initialize_assignments_to_random()

            // pseudo: initialize_centroids_to_random()

            initialized = true;
            node.callback();
        });
    });

    auto finalizer = noarr::structures::compute_node([&](builder node){
        auto points = node.link(points_hub.read(Device::HOST_INDEX));
        auto assignments = node.link(assignments_hub.read(Device::HOST_INDEX));
        auto centroids = node.link(centroids_hub.read(Device::HOST_INDEX));

        node.can_advance([&](){
            return true; // conditioned by having a chunk of data to process
        });

        node.advance([&](){
            // pseudo: write_envelope_values_to_output_arguments()
            // for (point_t p in points.envelope.buffer)
            //     computed_centroids.push_back(p)

            node.callback();
        });
    });

    auto iterator = noarr::structures::cuda_compute_node([&](builder node) {
        auto points = node.link(points_hub.readwrite(Device::DEVICE_INDEX));
        auto assignments = node.link(assignments_hub.readwrite(Device::DEVICE_INDEX));
        auto centroids = node.link(centroids_hub.readwrite(Device::DEVICE_INDEX));

        node.can_advance([&](){
            return initialized && next_iteration < refinements;
        });

        node.advance([&](){
            // pseudo: update_centroids_kernel(points, centroids)
            // pseudo: assign_centroids_kernel(points, assignments, centroids)

            cudaSynchronize(node.cuda_stream, node.calback);
        });

        node.finalize([&](){
            ++next_iteration;

            // after last iteration swith dataflow back to the host
            if (next_iteration >= refinements) {
                points_hub.set_dataflow_strategy.to_link(finalizer);
                assignments_hub.set_dataflow_strategy.to_link(finalizer);
                centroids_hub.set_dataflow_strategy.to_link(finalizer);
            }
        });
    });

    // at the beginning, move the latest data to the device
    points_hub.set_dataflow_strategy.to_link(iterator);
    assignments_hub.set_dataflow_strategy.to_link(iterator);
    centroids_hub.set_dataflow_strategy.to_link(iterator);

    // setup scheduler
    auto sched = noarr::pipelines::scheduler();
    sched.add(points_hub);
    sched.add(assignments_hub);
    sched.add(centroids_hub);
    sched.add(initializer);
    sched.add(iterator);
    sched.add(finalizer);

    // run
    sched.run();
}
```
