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
        this->__internal__initialize();
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
        if (!this->__internal__can_advance())
        {
            scheduler_callback(false);
            return;
        }

        // if the data can be advanced, do it
        advance_callback = [this, &scheduler_callback](){
            advance_callback = nullptr;
            scheduler_callback(true); // data has been advanced
        };
        this->__internal__advance();
    }

    /**
     * Called by the scheduler after an update finishes.
     * Guaranteed to run on the scheduler thread.
     */
    void scheduler_post_update(bool data_was_advanced) {
        if (data_was_advanced)
            this->__internal__post_advance();
    }

    /**
     * Called by the scheduler after the pipeline finishes running.
     */
    void scheduler_terminate() {
        this->__internal__terminate();
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
            "Callback can be called only once and only after the 'advance' method has started"
        );

        advance_callback();
    }

    /**
     * Returns true, if the "callback()" method can be called.
     * Anotherwords the "advance" method has started and "callback"
     * hasn't been called yet.
     */
    bool can_call_callback() {
        return advance_callback != nullptr;
    }

    ///////////////////////
    // Internal Node API //
    ///////////////////////

    /*
        Represents the same concepts as the external node API,
        but is meant to be used by classes that extend the logic
        (e.g. ComputeNode, AsyncComputeNode, ...)

        The end user might override can_advance, for example, and forget to
        call the base class implementation. Which might break the base class
        logic if it needs to act during this event. Therefore the internal API
        is meant for internal node logic and it MUST CALL PARENT IMPLEMENTATIONS.
        Whereas the external API are the final methods called and they don't need
        to wory about inheritance.
    */

protected:

    virtual void __internal__initialize() {
        initialize();
    }

    virtual bool __internal__can_advance() {
        return can_advance();
    }

    virtual void __internal__advance() {
        advance();
    }

    virtual void __internal__post_advance() {
        post_advance();
    }

    virtual void __internal__terminate() {
        terminate();
    }

    ///////////////////////
    // External Node API //
    ///////////////////////

    /*
        The following event functions are meant to be overriden by the end user.
    */

protected:

    /**
     * Called before anything starts happening with the node
     */
    virtual void initialize() {
        //
    }

    /**
     * Called to test, whether the advance method can be called
     */
    virtual bool can_advance() {
        return false;
    }

    /**
     * Called to advance the proggress of data through the node
     */
    virtual void advance() {
        this->callback();
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
