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
            //
        }
    }

private:
    std::vector<std::unique_ptr<pipe_envelope>> envelopes;
    std::vector<std::unique_ptr<pipe_compute_node>> nodes;

    std::unique_ptr<compute_node> terminal_node;
};

} // namespace pipelines
} // namespace noarr

#endif
