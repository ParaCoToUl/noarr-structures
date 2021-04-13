#ifndef NOARR_PIPELINES_HARBOR_HPP
#define NOARR_PIPELINES_HARBOR_HPP

#include <cstddef>
#include <functional>
#include "untyped_dock.hpp"

namespace noarr {
namespace pipelines {

class harbor {
public:

    /**
     * Called by the scheduler before the pipeline starts running
     */
    void scheduler_start() {
        this->perform_dock_registration();
    }

    /**
     * Called by the scheduler when possible,
     * guaranteed to not be called when another
     * update of this harbor is still running
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

        this->send_ships();
    }

protected:

    /**
     * This function is called before the pipeline starts
     * to register all harbor docks
     */
    virtual void register_docks(std::function<void(untyped_dock*)> register_dock) = 0;

    /**
     * Called to test, whether the advance method can be called
     */
    virtual bool can_advance() = 0;

    /**
     * Called to advance the proggress of data through the harbor
     */
    virtual void advance(std::function<void()> callback) = 0;

    /**
     * Called after the data advancement
     */
    virtual void post_advance() {
        //
    }

private:

    std::vector<untyped_dock*> registered_docks;

    /**
     * Calls the register_docks method
     */
    void perform_dock_registration() {
        this->registered_docks.clear();
        
        this->register_docks([&](untyped_dock* d) {
            this->registered_docks.push_back(d);
        });
    }

    /**
     * Sends ships that are ready to leave
     */
    void send_ships() {
        for (untyped_dock* d : this->registered_docks)
        {
            // TODO: move untyped_dock::state to dock_state
            if (d->get_state() != untyped_dock::state::processed)
                continue;

            if (d->ship_target == nullptr)
                continue;

            if (d->ship_target->get_state() != untyped_dock::state::empty)
                continue;

            untyped_ship* s = d->docked_ship;
            d->docked_ship = nullptr;
            d->ship_target->arrive_ship(s);
        }
    }
};

} // pipelines namespace
} // namespace noarr

#endif
