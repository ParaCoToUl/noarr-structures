#ifndef NOARR_PIPELINES_SCHEDULER_HPP
#define NOARR_PIPELINES_SCHEDULER_HPP

#include <cstddef>
#include <exception>
#include "Node.hpp"

namespace noarr {
namespace pipelines {

/**
 * Scheduler manages the runtime of a pipeline
 */
class scheduler {
public:

    /**
     * Registers a node to be updated by the scheduler
     */
    void add(Node* h) {
        this->nodes.push_back(h);
    }

    /**
     * Runs the pipeline until no nodes can be advanced.
     */
    void run() {
        
        // call the start method on each node
        for (Node* h : this->nodes)
            h->scheduler_start();
        
        /*
            A naive scheduler implementation with no parallelism.
            
            TODO: parallelize memory transfers, device computation
                and host computation
         */

        bool some_node_was_advanced = true;
        do
        {
            some_node_was_advanced = false;

            for (Node* h : this->nodes)
            {
                bool data_was_advanced;

                this->callback_will_be_called();

                h->scheduler_update([&](bool adv){
                    data_was_advanced = adv;
                    this->callback_was_called();
                });

                this->wait_for_callback();

                h->scheduler_post_update(data_was_advanced);
                
                if (data_was_advanced)
                    some_node_was_advanced = true;
            }
        }
        while (some_node_was_advanced);
    }

private:

    /**
     * Nodes that the scheduler periodically updates
     */
    std::vector<Node*> nodes;

    ///////////////////////////
    // Synchronization logic //
    ///////////////////////////

    // This is a dummy implementation that only supports synchronous nodes.
    // TODO: Add a proper synchronization primitive on which the scheduler
    // thread can wait and let the callback_was_called method (or its
    // quivalent) be callable from any thread.

    bool _expecting_callback;
    bool _callback_was_called;

    /**
     * Call this before starting a node update
     */
    void callback_will_be_called() {
        if (this->_expecting_callback)
            throw std::runtime_error(
                "Cannot expect a callback when the previous didn't finish."
            );

        this->_expecting_callback = true;
        this->_callback_was_called = false;
    }

    /**
     * Call this from the node callback
     */
    void callback_was_called() {
        this->_callback_was_called = true;
    }

    /**
     * Call this to synchronously wait for the callback
     */
    bool wait_for_callback() {
        if (!this->_expecting_callback)
            throw std::runtime_error(
                "Cannot wait for callback without first expecting it."
            );

        if (!this->_callback_was_called)
            throw std::runtime_error(
                "TODO: Asynchronous nodes are not implemented yet."
            );

        this->_expecting_callback = false;
    }

};

} // pipelines namespace
} // namespace noarr

#endif
