Envelopes with in-place ship modification.

```cpp
auto loader = my_loader();
auto loader_envelope = envelope().write_dock(HOST).read_dock(DEVICE);
auto processor = my_processor();
auto processor_envelope = envelope().write_dock(DEVICE).read_dock(HOST);
auto printer = my_printer();

loader.output_dock.ship_data_to(loader_envelope.write_dock);
processor.data_dock.ship_data_from(loader_envelope.read_dock);
processor.data_dock.ship_data_to(processor_envelope.write_dock);
processor_envelope.ships(DEVICE_SHIPS).send_empty_ships_to(loader_envelope);

auto scheduler = the_scheduler();
scheduler.add(loader);
scheduler.add(loader_envelope);
scheduler.add(processor);
scheduler.add(processor_envelope);
scheduler.add(printer);
scheduler.run();
```


# Ships, docks and harbors

The true low-level interface.

Docks can have a ship. It also has a flag `processed and ready to leave`.
~~Scheduler is responsible~~ Harbor during finalization is responsible for
ship movement between docks, because there may be two docks sending ships
to one dock, but you cannot have two ships arrive at one dock simultaneously.

A dock doesn't care about the source of ships, but it knows where to send
ships that have been handled. Harbors can take ships from docks inside or
put ships from inside to docks. Ship inside a harbor is completely in its
management, can be relocated to a different dock, deleted or new one created.

```cpp
// en example with one producer and one consumer directly linked

class my_producer : public harbor {
public:
    dock output_dock;

    bool can_advance() override {
        return this->output_dock.has_ship && !this->output_dock.ready_to_leave;
    }

    // parent method
    void scheduler_poke() {
        if (this->can_advance())
            return;
        
        // TODO: asynchronous
        this->being_advanced = true;
        this->advance();
        this->being_advanced = false;
    }

    void advance() override {
        auto s = this->output_dock.ship;
        s.structure = $$$;
        s.data = $$$;

        this->output_dock.ready_to_leave = true;
    }
};

// TODO: my_consumer

// create "compute nodes"
auto producer = my_producer();
auto consumer = my_consumer();

// link docks together
producer.output_dock.send_processed_ships_to(consumer.input_dock)
consumer.input_dock.send_processed_ships_to(producer.output_dock)

// create the first ship that will cycle between the nodes
auto my_ship = ship(...);
producer.output_dock.arrive_ship(my_ship);

// setup scheduler
// ...
```


# Compute nodes and envelopes

Envelopes abstract away ship creation and host-device data transfer.

```cpp
// an example with one producer and one consumer on the same device
// with one envelope in between

// create the envelope
auto env = envelope(SAME_DEVICE, BUFFERS_1, SOMETHING);

// create compute nodes
auto producer = my_producer(env);
auto consumer = my_consumer(env);

// setup scheduler
// ...
```












# Example program to ...

Jenom lehkej wrapper kolem cudy....

```cpp
void kmeans(const Points points, std::size_t k, std::size_t refinements) {
    Envelope<...> pts;
    Envelope<...> assignments;
    Envelope<...> centroids;

    pts.initialize_from(points, cudaStream);

    centroids = ....;

    for (refinemets) {
        updateCentroidsKernel(pts, assignments, centroids);
        assignCentroidsKernel(pts, assignments, centroids);
    }

    cudaSynchronize(cudaStream);
}

void main() {

}
```



# K-means

```cpp
struct point_t {
    float x, y;
}


///////////////
// Attempt 1 //
///////////////

void kmeans(
    const std::vector<point_t> given_points,
    std::size_t k,
    std::size_t refinements
) {
    Hub<std::size_t, point_t> points_hub;
    Hub<std::size_t, std::uint8> assignments_hub;
    Hub<std::size_t, point_t> centroids_hub;

    bool initialized = false;

    auto initializer = noarr::structures::compute_node()
        .link(points_hub.write(Device::HOST_INDEX))
        .link(assignments_hub.write(Device::HOST_INDEX))
        .link(centroids_hub.write(Device::HOST_INDEX))
        .can_advance([&](){
            return !initialized;
        })
        .advance([&](
            callback callback,
            Link<std::size_t, point_t> points,
            Link<std::size_t, std::uint8> assignments,
            Link<std::size_t, point_t> centroids,
        ){
            points.envelope.structure = given_points.size();
            for (std::size_t i = 0; i < given_points.size(); ++i)
                points.envelope.buffer[i] = given_points[i];

            // pseudo: initialize_assignments_to_random()

            // pseudo: initialize_centroids_to_random()

            initialized = true;
            callback();
        });

    // at the beginning, move the latest data to the device
    points_hub.set_dataflow_strategy.to_link(the_write_link_variable);

    // OLD code:

    centroids = ....;

    for (refinemets) {
        updateCentroidsKernel(pts, assignments, centroids);
        assignCentroidsKernel(pts, assignments, centroids);
    }

    cudaSynchronize(cudaStream);
}


///////////////
// Attempt 2 //
///////////////

void kmeans(
    const std::vector<point_t> given_points,
    std::size_t k,
    std::size_t refinements
) {
    //auto pipeline = noarr::pipelines::pipeline;

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
            return true;
        });

        node.advance([&](){
            // pseudo: write_envelope_values_to_output_arguments()

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
