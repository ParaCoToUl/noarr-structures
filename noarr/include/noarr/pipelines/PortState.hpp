#ifndef NOARR_PIPELINES_PORT_STATE_HPP
#define NOARR_PIPELINES_PORT_STATE_HPP

namespace noarr {
namespace pipelines {

/**
 * Possible states of a node port
 */
enum PortState : unsigned char {
    
    /**
     * No envelope attached
     */
    empty = 0,

    /**
     * A envelope has been attached but hasn't been processed yet
     */
    arrived = 1,

    /**
     * The envelope has been processed and is ready to leave
     */
    processed = 2
};

} // pipelines namespace
} // namespace noarr

#endif
