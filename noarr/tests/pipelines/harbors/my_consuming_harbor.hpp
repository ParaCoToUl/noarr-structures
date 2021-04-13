#include <string>
#include <functional>

#include <noarr/pipelines/dock.hpp>
#include <noarr/pipelines/untyped_dock.hpp>
#include <noarr/pipelines/harbor.hpp>

using namespace noarr::pipelines;

class my_consuming_harbor : public harbor {
public:
    dock<std::size_t, char> input_dock;

    std::string received_string;

    my_consuming_harbor() {
        this->received_string.clear();
    }

    virtual void register_docks(std::function<void(untyped_dock*)> register_dock) {
        register_dock(&this->input_dock);
    };

    bool can_advance() override {
        // true, if we have a full ship available
        return this->input_dock.get_state() == untyped_dock::state::arrived;
    }

    void advance(std::function<void()> callback) override {
        // get the ship to be filled up
        auto& ship = this->input_dock.get_ship();

        // move the chunk from ship into the accumulator
        this->received_string.append(ship.buffer, ship.structure);
        ship.has_payload = false;
        this->input_dock.ship_processed = true;

        // computation is done
        callback();
    }
};
