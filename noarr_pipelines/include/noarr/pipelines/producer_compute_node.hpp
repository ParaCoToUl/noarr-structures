#ifndef NOARR_PIPELINES_PRODUCER_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_PRODUCER_COMPUTE_NODE_HPP 1

#include "noarr/pipelines/envelope.hpp"
#include "noarr/pipelines/compute_node.hpp"

namespace noarr {
namespace pipelines {

/**
 * A compute node with one output port
 */
template<
    typename OutputStructure,
    typename OutputBufferItem = void
>
class producer_compute_node : public virtual compute_node {
public:
    using output_port_t = envelope::port<OutputStructure, OutputBufferItem>;

    void set_output_port(output_port_t p) {
        this->output_port = p;
    };

protected:
    output_port_t output_port;
};

} // pipelines namespace
} // namespace noarr

#endif
