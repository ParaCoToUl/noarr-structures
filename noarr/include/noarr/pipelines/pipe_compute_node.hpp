#ifndef NOARR_PIPELINES_PIPE_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_PIPE_COMPUTE_NODE_HPP 1

#include "noarr/pipelines/consumer_compute_node.hpp"
#include "noarr/pipelines/producer_compute_node.hpp"

namespace noarr {
namespace pipelines {

/**
 * A compute node with one input port and one output port
 */
template<
    typename InputStructure,
    typename OutputStructure,
    typename InputBufferItem = void,
    typename OutputBufferItem = void
>
class pipe_compute_node :
    public consumer_compute_node<InputStructure, InputBufferItem>,
    public producer_compute_node<OutputStructure, OutputBufferItem>
{
    //
};

} // pipelines namespace
} // namespace noarr

#endif
