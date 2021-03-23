#ifndef NOARR_PIPELINES_CONSUMER_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_CONSUMER_COMPUTE_NODE_HPP 1

#include "noarr/pipelines/envelope.hpp"
#include "noarr/pipelines/compute_node.hpp"

namespace noarr {
namespace pipelines {

/**
 * A compute node with one input port
 */
template<
    typename InputStructure,
    typename InputBufferItem = void
>
class consumer_compute_node : public virtual compute_node {
public:
    using input_port_t = envelope::port<InputStructure, InputBufferItem>;

    void set_input_port(input_port_t p) {
        this->input_port = p;
    }

protected:
    input_port_t input_port;
};

} // pipelines namespace
} // namespace noarr

#endif
