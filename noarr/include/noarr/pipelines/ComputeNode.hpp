#ifndef NOARR_PIPELINES_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_COMPUTE_NODE_HPP

#include <cstddef>
#include <vector>
#include <iostream>

#include "noarr/pipelines/Node.hpp"
#include "noarr/pipelines/Link.hpp"

namespace noarr {
namespace pipelines {

/**
 * Represents a node that performs some computation
 */
class ComputeNode : public Node {
public:

    ComputeNode() : Node() { }
    ComputeNode(const std::string& label) : Node(label) { }

    /**
     * All the links attached to this compute node
     */
    std::vector<UntypedLink*> links;

    /**
     * Connects a link
     */
    template<typename TLink>
    TLink& link(TLink& link) {
        links.push_back(&link);
        link.set_guest_node(this);
        return link;
    }

    bool are_links_ready() {
        for (UntypedLink* link : links) {
            if (link->untyped_envelope == nullptr)
                return false;

            if (link->state != LinkState::fresh)
                return false;
        }
        
        return true;
    }

    void finalize_links_after_advance() {
        for (UntypedLink* link : links) {
            if (link->autocommit && !link->was_committed)
                link->commit();

            link->state = LinkState::processed;
            link->call_callback();
        }
    }

protected:

    virtual bool __internal__can_advance() {
        // a compute node cannot start, unless it has envelopes on all links
        if (!are_links_ready())
            return false;
        
        // then do the usual "can_advance" logic
        return Node::__internal__can_advance();
    }

    virtual bool can_advance() override {
        // NOTE: generic node has default to false,
        // but compute node has default to true, because it is conditioned by links
        return true;
    }

    /**
     * Wrapper around post_advance that should alway call its base
     * implementation. Exists only so that the final user can override
     * can_advance and forget to call the base implementation and not get roasted.
     */
    virtual void __internal__post_advance() override {
        Node::__internal__post_advance(); // call "post_advance"

        finalize_links_after_advance();
    }
};

} // pipelines namespace
} // namespace noarr

#endif
