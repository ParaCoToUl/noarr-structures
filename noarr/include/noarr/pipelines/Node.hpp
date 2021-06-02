#ifndef NOARR_PIPELINES_NODE_HPP
#define NOARR_PIPELINES_NODE_HPP

#include <string>
#include <functional>

namespace noarr {
namespace pipelines {

class Node {
public:
    /**
     * Label that can be used in logging and error messages
     */
    std::string label;

    Node() : label(std::to_string((unsigned long)this)) { }

    Node(const std::string& label) : label(label) { }

    ///////////////////
    // Scheduler API //
    ///////////////////

private:
    std::function<void()> advance_callback = nullptr;

public:

    /**
     * Called by the scheduler before the pipeline starts running.
     */
    void scheduler_initialize() {
        this->initialize();
    }

    /**
     * Called by the scheduler as often as possible.
     * 
     * Guaranteed to not be called when another update of this same
     * node instance is still running.
     * 
     * @param callback Call when the async operation finishes.
     * Pass true if data was advanced and false if nothing happened.
     */
    void scheduler_update(std::function<void(bool)> scheduler_callback) {
        // if the data cannot be advanced, return immediately
        if (!this->wrapper_around_can_advance())
        {
            scheduler_callback(false);
            return;
        }

        // if the data can be advanced, do it
        advance_callback = [this, &scheduler_callback](){
            advance_callback = nullptr;
            scheduler_callback(true); // data has been advanced
        };
        this->advance();
    }

    /**
     * Called by the scheduler after an update finishes.
     * Guaranteed to run on the scheduler thread.
     */
    void scheduler_post_update(bool data_was_advanced) {
        if (data_was_advanced)
            this->wrapper_around_post_advance();
    }

    /**
     * Called by the scheduler after the pipeline finishes running.
     */
    void scheduler_terminate() {
        this->terminate();
    }

    //////////////
    // Node API //
    //////////////

public:

    /**
     * Call this from within the advance method
     * to signal the end of the computation
     */
    void callback() {
        assert(
            (advance_callback != nullptr) &&
            "Cannot call the callback when the node hasn't been advanced"
        );

        advance_callback();
    }

protected:

    /**
     * Called before anything starts happening with the node
     */
    virtual void initialize() {
        //
    }

    /**
     * Wrapper around can_advance that can do additional processing.
     * This wrapper should always call the paren't version and take account,
     * the can_advance need not to.
     */
    virtual bool wrapper_around_can_advance() {
        return can_advance();
    }

    /**
     * Called to test, whether the advance method can be called
     */
    virtual bool can_advance() = 0;

    /**
     * Called to advance the proggress of data through the node
     */
    virtual void advance() = 0;

    /**
     * Wrapper around post_advance that should alway call its base
     * implementation. Exists only so that the final user can override
     * can_advance and forget to call the base implementation and not get roasted.
     */
    virtual void wrapper_around_post_advance() {
        post_advance();
    }

    /**
     * Called after the data advancement
     */
    virtual void post_advance() {
        //
    }

    /**
     * Called after all the computation finishes
     */
    virtual void terminate() {
        //
    }
};

} // pipelines namespace
} // namespace noarr

#endif
