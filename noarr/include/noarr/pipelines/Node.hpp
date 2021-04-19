#ifndef NOARR_PIPELINES_NODE_HPP
#define NOARR_PIPELINES_NODE_HPP

#include <cstddef>
#include <functional>
#include "UntypedPort.hpp"

namespace noarr {
namespace pipelines {

class Node {
public:

    /**
     * Called by the scheduler before the pipeline starts running
     */
    void scheduler_start() {
        this->perform_port_registration();
    }

    /**
     * Called by the scheduler when possible,
     * guaranteed to not be called when another
     * update of this node is still running
     * 
     * @param callback Call when the async operation finishes,
     * pass true if data was advanced and false if nothing happened
     */
    void scheduler_update(std::function<void(bool)> callback) {
        // if the data cannot be advanced, return immediately
        if (!this->can_advance())
        {
            callback(false);
            return;
        }

        // if the data can be advanced, do it
        this->advance([&](){
            // data has been advanced
            callback(true);
        });
    }

    /**
     * Called by the scheduler after an update finishes.
     * Guaranteed to run on the scheduler thread.
     */
    void scheduler_post_update(bool data_was_advanced) {
        if (data_was_advanced)
            this->post_advance();

        this->send_envelopes();
    }

protected:

    /**
     * This function is called before the pipeline starts
     * to register all node ports
     */
    virtual void register_ports(std::function<void(UntypedPort*)> register_port) = 0;

    /**
     * Called to test, whether the advance method can be called
     */
    virtual bool can_advance() = 0;

    /**
     * Called to advance the proggress of data through the node
     */
    virtual void advance(std::function<void()> callback) = 0;

    /**
     * Called after the data advancement
     */
    virtual void post_advance() {
        //
    }

private:

    std::vector<UntypedPort*> registered_ports;

    /**
     * Calls the register_ports method
     */
    void perform_port_registration() {
        this->registered_ports.clear();
        
        this->register_ports([&](UntypedPort* d) {
            this->registered_ports.push_back(d);
        });
    }

    /**
     * Sends envelopes that are ready to leave
     */
    void send_envelopes() {
        for (UntypedPort* d : this->registered_ports)
        {
            if (d->get_state() != UntypedPort::state::processed)
                continue;

            if (d->envelope_target == nullptr)
                continue;

            if (d->envelope_target->get_state() != UntypedPort::state::empty)
                continue;

            UntypedEnvelope* env = d->attached_envelope;
            d->attached_envelope = nullptr;
            d->envelope_target->attach_envelope(env);
        }
    }
};

} // pipelines namespace
} // namespace noarr

#endif
