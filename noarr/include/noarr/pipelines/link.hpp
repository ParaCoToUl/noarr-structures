#ifndef NOARR_PIPELINES_LINK_HPP
#define NOARR_PIPELINES_LINK_HPP

#include <cstddef>

namespace noarr {
namespace pipelines {

/**
 * A link between an envelope and a compute node
 */
struct link {
public:

    /**
     * On what device this link exists
     * (-1 is the host)
     */
    char device_index = -1;

    /**
     * What state is the link in
     */
    char state = 0;
    const char STATE_NOT_READY = 0;
    const char STATE_READY = 1;
    const char STATE_IN_USE = 2;
    const char STATE_USED = 3;

    /**
     * Link flags
     */
    char flags = 0;
    const char READ = 1;
    const char WRITE = 2;
    const char EXCLUSIVE = 4;
};

} // pipelines namespace
} // namespace noarr

#endif
