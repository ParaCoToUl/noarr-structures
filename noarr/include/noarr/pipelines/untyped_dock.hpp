#ifndef NOARR_PIPELINES_UNTYPED_DOCK_HPP
#define NOARR_PIPELINES_UNTYPED_DOCK_HPP

#include <cstddef>
#include <exception>
#include "memory_device.hpp"
#include "untyped_ship.hpp"

namespace noarr {
namespace pipelines {

class untyped_dock {
public:

    /**
     * Possible states of the dock
     */
    enum state : unsigned char {
        /**
         * No ship at the dock
         */
        empty = 0,

        /**
         * A ship has arrived but hasn't been processed yet
         */
        arrived = 1,

        /**
         * The ship has been processed and is ready to leave
         */
        processed = 2
    };

    /**
     * Returns the state of the dock
     */
    state get_state() {
        if (this->docked_ship == nullptr)
            return state::empty;
        if (this->ship_processed)
            return state::processed;
        return state::arrived;
    }

    /**
     * Set the target dock, to which processed ships are sent
     */
    void send_processed_ships_to(untyped_dock* target) {
        this->ship_target = target;
    }

    /**
     * Returns a reference to the docked ship
     */
    untyped_ship& get_untyped_ship() {
        if (this->docked_ship == nullptr)
            throw std::runtime_error("Cannot get a ship when none is docked.");

        return *this->docked_ship;
    }

    /**
     * Perform a ship arrival to this dock
     */
    void arrive_ship(untyped_ship* ship) {
        if (this->docked_ship != nullptr)
            throw std::runtime_error("There's a ship already present.");

        // TODO: overload the equality operator?
        if (ship->device.device_index != this->device.device_index)
            throw std::runtime_error("The ship belongs to a different device.");

        this->docked_ship = ship;
        this->ship_processed = false;
    }

    /**
     * The device on which the dock exists
     */
    memory_device device;

    untyped_ship* docked_ship = nullptr;
    
    bool ship_processed = false;

private:
    untyped_dock* ship_target = nullptr;
};

} // pipelines namespace
} // namespace noarr

#endif
