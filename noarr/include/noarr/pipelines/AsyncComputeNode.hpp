#ifndef NOARR_PIPELINES_ASYNC_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_ASYNC_COMPUTE_NODE_HPP

#include <cstddef>
#include <vector>
#include <iostream>
#include <functional>
#include <thread>

#include "noarr/pipelines/ComputeNode.hpp"

namespace noarr {
namespace pipelines {

/**
 * Asynchronous compute node runs additional function "advance_async"
 * in a background thread
 */
class AsyncComputeNode : public ComputeNode {
public:
    AsyncComputeNode() : ComputeNode() { }
    AsyncComputeNode(const std::string& label) : ComputeNode(label) { }

    void __internal__advance() override {
        ComputeNode::__internal__advance(); // call the standard "advance"

        // callback was already called in the "advance" method,
        // so we don't need to continue to "advance_async"
        if (!this->can_call_callback())
            return;

        // now, start the background magic
        std::thread background_thread(
            &AsyncComputeNode::__internal__advance_async,
            this
        );
        background_thread.detach();
    }

    virtual void __internal__advance_async() {
        this->advance_async();

        // the user need not call the callback inside the advance_async method,
        // they can just return
        // (this is because we don't expect any additional asynchronous jobs
        // to be started and if so, the user should .join() on them before returning)
        if (this->can_call_callback())
            this->callback();
    }

    /**
     * Called right after "advance" with the same goal of advancing data,
     * but it runs in a background thread -- not blocking the scheduler thread.
     */
    virtual void advance_async() {
        this->callback();
    }
};

} // pipelines namespace
} // namespace noarr

#endif
