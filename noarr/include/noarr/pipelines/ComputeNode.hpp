#ifndef NOARR_PIPELINES_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_COMPUTE_NODE_HPP

#include <cstddef>
#include <vector>
#include <iostream>

#include "noarr/pipelines/Node.hpp"

namespace noarr {
namespace pipelines {

/**
 * Represents a node that performs some computation
 */
class ComputeNode : public Node {
public:

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
    /**
     * Called before anything starts happening with the node
     */
    virtual void initialize() override {
        //
    }

    /**
     * Wrapper around can_advance that can do additional processing.
     * This wrapper should always call the paren't version and take account,
     * the can_advance need not to.
     */
    virtual bool wrapper_around_can_advance() {
        // a compute node cannot start, unless it has envelopes on all links
        if (!are_links_ready())
            return false;
        
        // then do the usual "can_advance" logic
        return Node::wrapper_around_can_advance();
    }

    /**
     * Called before advancement to check the data can be advanced
     */
    virtual bool can_advance() override {
        return true;
    }

    /**
     * Called on the scheduler thread to advance the data processing
     */
    virtual void advance() override {
        this->callback();
    }

    /**
     * Wrapper around post_advance that should alway call its base
     * implementation. Exists only so that the final user can override
     * can_advance and forget to call the base implementation and not get roasted.
     */
    virtual void wrapper_around_post_advance() override {
        Node::wrapper_around_post_advance(); // call "post_advance"

        finalize_links_after_advance();
    }

    /**
     * Called after advancement on the scheduler thread
     */
    virtual void post_advance() override {
        //
    }

    /**
     * Called after all the computation finishes
     */
    virtual void terminate() override {
        //
    }
};

} // pipelines namespace
} // namespace noarr

#endif
