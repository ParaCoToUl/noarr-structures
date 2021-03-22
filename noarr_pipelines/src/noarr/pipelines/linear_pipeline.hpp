#ifndef NOARR_PIPELINES_LINEAR_PIPELINE_HPP
#define NOARR_PIPELINES_LINEAR_PIPELINE_HPP 1

#include <memory>
#include <vector>
#include "noarr/pipelines/envelope.hpp"

namespace noarr {
namespace pipelines {

/**
 * Simplifies the process of building out a linear pipeline
 */
class linear_pipeline {
public:

    // NOTE: immutable type-changing builder object that will join nodes and envelopes

    /**
     * Run the linear pipeline to completion
     */
    void run() {
        // TODO: verify the pipeline has been fully built

        // execute workers until the final compute node
        // has received end of stream
        // TODO ...

        while (!this->terminal_node->has_processed_all_chunks())
        {
            for (chunk_stream_processor processor : chunk_stream_processors)
            {
                if (processor.is_ready_for_next_chunk())
                    processor.start_next_chunk_processing();
            }

            thread.yield();
        }
    }

private:
    std::vector<std::unique_ptr<pipe_envelope>> envelopes;
    std::vector<std::unique_ptr<pipe_compute_node>> nodes;

    std::unique_ptr<compute_node> terminal_node;
};






void foo() {

    linear_pipeline my_pipeline;

    my_pipeline.run();

    auto m = my_mapping_node();

    liner_pipeline(
        my_loader_node(),
        pipe_envelope(h2d, true),
        my_reducing_node(buffer_envelope())
        pipe_envelope(d2h, true),
        my_printer_node()
    ).run();

}



} // namespace pipelines
} // namespace noarr

#endif
