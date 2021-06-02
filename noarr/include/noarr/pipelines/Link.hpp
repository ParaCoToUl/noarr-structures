#ifndef NOARR_PIPELINES_LINK_HPP
#define NOARR_PIPELINES_LINK_HPP

#include <cstddef>
#include <functional>

#include "noarr/pipelines/Envelope.hpp"

namespace noarr {
namespace pipelines {

enum LinkType : unsigned char {
    /**
     * Produces new chunks of data
     */
    producing = 1,

    /**
     * Consumes chunks of data
     */
    consuming = 2,

    /**
     * Reads existing chunks of data (without modification)
     */
    peeking = 4,

    /**
     * Modifies existing chunks of data
     */
    modifying = 6,
};

enum LinkState : unsigned char {
    /**
     * The envelope (if attached) has not yet been processed by the guest
     */
    fresh = 0,

    /**
     * The envelope has been processed by the guest and can be removed by the host
     */
    processed = 1,
};

class UntypedLink { // abstract class
public:
    /**
     * Type of this link
     */
    LinkType type;

    /**
     * State of this link
     */
    LinkState state = LinkState::fresh;

    /**
     * What device this link lives on
     */
    Device::index_t device_index;

    /**
     * Producing and consuming operations may or may not produce or consume
     * the chunk, when not done explicitly, this flag forces it when set
     */
    bool autocommit;

    /**
     * Whether the link was commited by the guest or not (during processing)
     */
    bool was_committed = false;

    /**
     * Called when the envelope is processed
     */
    std::function<void()> callback = nullptr;

    /**
     * The envelope this link provides access to, in the untyped form
     */
    UntypedEnvelope* untyped_envelope = nullptr;

    UntypedLink(
        LinkType type,
        Device::index_t device_index,
        bool autocommit
    ) :
        type(type),
        device_index(device_index),
        autocommit(autocommit)
    {}

    /**
     * Host commits the link
     */
    void commit() {
        was_committed = true;
    }

    /**
     * Calls the link callback. Call this when the envelope has been processed.
     */
    void call_callback() {
        assert((callback != nullptr) && "Link callback is set to null");
        callback();
    }
};

/**
 * Links two nodes so that the guest can access envelopes on the host
 */
template<typename Structure, typename BufferItem = void>
class Link : public UntypedLink {
public:
    /**
     * The envelope this link provides access to
     */
    Envelope<Structure, BufferItem>* envelope;

    Link(
        LinkType type,
        Device::index_t device_index,
        bool autocommit
    ) : UntypedLink(type, device_index, autocommit), envelope(nullptr) { }

    /**
     * The host provides an envelope to the link for the guest to access
     */
    void host_envelope(
        Envelope<Structure, BufferItem>& envelope,
        std::function<void()> callback
    ) {
        this->envelope = &envelope;
        this->untyped_envelope = &envelope;
        this->callback = callback;
        this->state = LinkState::fresh;
        this->was_committed = false;
    }

    /**
     * Detaches the envelope from the link after it has been processed
     */
    void detach_envelope() {
        this->envelope = nullptr;
        this->untyped_envelope = nullptr;
        this->callback = nullptr;
        this->state = LinkState::fresh;
        this->was_committed = false;
    }
};

} // pipelines namespace
} // namespace noarr

#endif
