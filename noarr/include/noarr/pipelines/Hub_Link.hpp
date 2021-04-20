#ifndef NOARR_PIPELINES_HUB_LINK_HPP
#define NOARR_PIPELINES_HUB_LINK_HPP

#include <functional>

#include "UntypedPort.hpp"
#include "Port.hpp"

namespace noarr {
namespace pipelines {

// forward declaration of a hub
template<typename Structure, typename BufferItem = void>
class Hub;

namespace hub {

enum LinkFlags : unsigned char {
    // r,w,p = 1,2,4

    // allowed
    read = 1,
    write = 2,
    readwrite = 3,
    read_peek = 5,
    readwrite_peek = 7,

    // elementary peek
    _peek = 4,
};

/**
 * A link between a Hub and a ComputeNode
 */
template<typename Structure, typename BufferItem = void>
class Link {
public:
    using attachment_implementation_t = std::function<
        void(Link<Structure, BufferItem>&, Port<Structure, BufferItem>&)
    >;

private:
    /**
     * What type of link is this
     */
    LinkFlags _flags;

    /**
     * What device this link lives on
     */
    Device::index_t _device_index;

    /**
     * Where to send processed envelopes
     * (the specific port depends on the flags)
     */
    Hub<Structure, BufferItem>& _forwarding_target;
    
    /**
     * Is the link already attached to a port?
     */
    bool _attached = false;

    /**
     * Function that performs the attachment, finalizing the link construction
     */
    attachment_implementation_t _attachment_implementation;

public:
    Link(
        Hub<Structure, BufferItem>& creator,
        LinkFlags flags,
        Device::index_t device_index,
        attachment_implementation_t attachment_implementation
    ) :
        _flags(flags),
        _device_index(device_index),
        _forwarding_target(creator),
        _attachment_implementation(attachment_implementation) { }

    /**
     * Getter for link flags
     */
    LinkFlags flags() const {
        return this->_flags;
    }

    /**
     * Getter for the processed envelope forwarding target
     */
    Hub<Structure, BufferItem>& forwarding_target() const {
        return this->_forwarding_target;
    }

    /**
     * Getter for the device index, this link lives on
     */
    Device::index_t device_index() const {
        return this->_device_index;
    }

    /**
     * Redirects processed envelopes to a different hub
     */
    Link& forward_processed_envelopes_to(Hub<Structure, BufferItem>& h) {
        assert((this->flags == LinkFlags::read || this->flags == LinkFlags::readwrite)
            && "Only 'read' or 'readwrite' links can forward envelopes");
        
        this->_forwarding_target = h;

        return *this;
    }

    /**
     * Attaches the non-hub end of the link to a given node port
     */
    Link& attach_to_port(Port<Structure, BufferItem>& port) {
        assert(!this->_attached && "The link is already attached");
        
        this->_attachment_implementation(*this, port);
        this->_attached = true;

        return *this;
    }
};

} // hub namespace
} // pipelines namespace
} // namespace noarr

#endif
