#ifndef NOARR_PIPELINES_ENVELOPE_HPP
#define NOARR_PIPELINES_ENVELOPE_HPP

#include <cstddef>
#include "noarr/pipelines/link.hpp"

namespace noarr {
namespace pipelines {

/**
 * Envelope represents a logical buffer
 */
template<typename Structure, typename BufferItem = void>
class envelope {
private:
    std::vector<std::unique_ptr<link>> links;
    std::size_t buffer_size;

public:
    envelope(std::size_t buffer_size) {
        this->buffer_size = buffer_size;
    }

    link* create_link() {
        this->links.push_back(std::make_unique<link>());
    }

    /**
     * Called by the scheduler to advance data through the envelope
     * @returns True if an asynchronous operation has been started
     */
    bool advance() {

        // go over all the links and find those that could be made ready

        // check these resulting links can be made ready in parallel

        // start routines that turn these links to ready and return the future
        
        return false;
    }

    void can_be_made_ready(link* l) {

    }

};

} // pipelines namespace
} // namespace noarr

#endif
