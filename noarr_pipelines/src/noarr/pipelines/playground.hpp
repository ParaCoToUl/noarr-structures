#include <cstddef>
#include "noarr/pipelines.hpp"

using namespace noarr::pipelines;

class my_producing_node : public producer_compute_node<std::size_t, int> {
private:
    std::vector<int>* items;
    std::size_t at;

public:
    my_producing_node(std::vector<int>* items) {
        this->items = items;
        this->at = 0;
    }

    bool is_ready_for_next_chunk() override {
        return ! this->output_port.contains_chunk();
    }

    void start_next_chunk_processing() override {
        int* output_buffer = this->output_port.get_buffer();
        std::size_t at_before = this->at;
        for (int i = 0; i < 3 && this->at < this->items->size(); i++) {
            output_buffer[i] = (*this->items)[this->at];
            this->at++;
        }
        this->output_port.set_structure(this->at - at_before);

        // TODO: maybe also distinguish "being filled up / being consumed" states?
        this->output_port.set_contains_chunk(true);

        if (this->at == this->items->size() - 1) {
            this->output_port.send_end_of_stream();
            this->set_all_chunks_processed();
        }

        this->set_chunk_processing_finished();
    }
};

class my_mapping_node : public pipe_compute_node<
    std::size_t, std::size_t, int, int
> {

    bool is_ready_for_next_chunk() override {
        return this->input_port.contains_chunk();
    }

    void start_next_chunk_processing() override {
        // === check end of stream ===
        if (this->input_port.contains_end_of_stream())
        {
            this->set_all_chunks_processed(); // we're done as a node
            this->output_port.send_end_of_stream(); // pass EOS downstream
            return;
        }

        // === perform the mapping operation ===

        // NOTE/TODO: here will be cast to a "bag" type
        std::size_t item_count = this->input_port.get_structure();
        int* input_buffer = this->input_port.get_buffer();
        int* output_buffer = this->output_port.get_buffer();

        // copy the number of items
        this->output_port.set_structure(item_count);

        // perform the map operation on the values
        for (int i = 0; i < item_count; i++)
            output_buffer[1 + i] = input_buffer[1 + i] * 2; // map = *2

        // the input buffer was consumed and the output buffer
        // was filled with a chunk of data
        this->input_port.set_contains_chunk(false);
        this->output_port.set_contains_chunk(true);

        // the asynchronous operation has finished
        // (well, it wasn't asnychronous at all in this case)
        this->set_chunk_processing_finished();
    }
};

class my_printing_node : public consumer_compute_node<std::size_t, int> {
public:
    std::string log;

    my_printing_node() {
        log.clear();
    }

    bool is_ready_for_next_chunk() override {
        return this->input_port.contains_chunk();
    }

    // TODO: separate external and internal API!
    void start_next_chunk_processing() override {
        // === check end of stream ===
        if (this->input_port.contains_end_of_stream())
        {
            this->set_all_chunks_processed();
            return;
        }

        // process the input chunk
        std::size_t item_count = this->input_port.get_structure();
        int* input_buffer = this->input_port.get_buffer();

        for (std::size_t i = 0; i < item_count; i++) {
            this->log.append(
                std::to_string(input_buffer[i])
            );
            this->log.append(";");
        }

        this->input_port.set_contains_chunk(false);

        this->set_chunk_processing_finished();
    }
};

void my_pipeline_running_function() {
    // prepare data that will go through the pipeline
    std::vector<int> items {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    // build the pipeline
    auto prod = my_producing_node(&items);

    auto env = move_h2d_envelope<std::size_t, std::size_t, int, int>();
    prod.set_output_port(env.get_input_port());

    auto print = my_printing_node();
    print.set_input_port(env.get_output_port());

    // run the pipeline to completion
    while (!print.has_processed_all_chunks())
    {
        if (prod.is_ready_for_next_chunk())
            prod.start_next_chunk_processing();

        if (env.is_ready_for_next_chunk())
            env.start_next_chunk_processing();

        if (print.is_ready_for_next_chunk())
            print.start_next_chunk_processing();
    }

    // print the result
    // print.log;
}
