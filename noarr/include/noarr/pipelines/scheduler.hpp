#ifndef NOARR_PIPELINES_SCHEDULER_HPP
#define NOARR_PIPELINES_SCHEDULER_HPP

#include <cstddef>
#include <exception>
#include "harbor.hpp"

namespace noarr {
namespace pipelines {

/**
 * Scheduler manages the runtime of a pipeline
 */
class scheduler {
public:

    /**
     * Registers a harbor to be updated by the scheduler
     */
    void add(harbor* h) {
        this->harbors.push_back(h);
    }

    /**
     * Runs the pipeline until no harbors can be advanced.
     */
    void run() {
        
        // call the start method on each harbor
        for (harbor* h : this->harbors)
            h->scheduler_start();
        
        /*
            A naive scheduler implementation with no parallelism.
            
            TODO: parallelize memory transfers, device computation
                and host computation
         */

        bool some_harbor_was_advanced = true;
        do
        {
            some_harbor_was_advanced = false;

            for (harbor* h : this->harbors)
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
                    some_harbor_was_advanced = true;
            }
        }
        while (some_harbor_was_advanced);
    }

private:

    /**
     * Harbors that the scheduler periodically updates
     */
    std::vector<harbor*> harbors;

    ///////////////////////////
    // Synchronization logic //
    ///////////////////////////

    // This is a dummy implementation that only supports synchronous harbors.
    // TODO: Add a proper synchronization primitive on which the scheduler
    // thread can wait and let the callback_was_called method (or its
    // quivalent) be callable from any thread.

    bool _expecting_callback;
    bool _callback_was_called;

    /**
     * Call this before starting a harbor update
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
     * Call this from the harbor callback
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
                "TODO: Asynchronous harbors are not implemented yet."
            );

        this->_expecting_callback = false;
    }

};

} // pipelines namespace
} // namespace noarr

#endif
