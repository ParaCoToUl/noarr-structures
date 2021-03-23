#ifndef NOARR_PIPELINES_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_COMPUTE_NODE_HPP 1

#include "noarr/pipelines/chunk_stream_processor.hpp"

namespace noarr {
namespace pipelines {

/**
 * Represents a node that performs some computation
 * (either on device or even on the host)
 * 
 * NOTE: Compared to envelopes, all compute nodes are async workers
 */
class compute_node : public chunk_stream_processor {
    // ...
};

} // pipelines namespace
} // namespace noarr

#endif
