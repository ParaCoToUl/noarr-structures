#include <string>
#include <functional>

#include <noarr/pipelines/dock.hpp>
#include <noarr/pipelines/untyped_dock.hpp>
#include <noarr/pipelines/harbor.hpp>

using namespace noarr::pipelines;

class my_producing_harbor : public harbor {
public:
    dock<std::size_t, char> output_dock;

    my_producing_harbor(std::string data, std::size_t chunk_size) {
        this->data = data;
        this->chunk_size = chunk_size;
        this->at_index = 0;
    }

    virtual void register_docks(std::function<void(untyped_dock*)> register_dock) {
        register_dock(&this->output_dock);
    };

    bool can_advance() override {
        // false, if we've processed the entire dataset
        if (this->at_index >= this->data.length())
            return false;
        
        // true, if we have an empty ship available
        return this->output_dock.get_state() == untyped_dock::state::arrived;
    }

    void advance(std::function<void()> callback) override {
        // get the ship to be filled up
        auto& ship = this->output_dock.get_ship();

        // compute the size of the next chunk
        std::size_t items_to_take = std::min(
            this->chunk_size,
            this->data.length() - this->at_index
        );

        // move the chunk onto the ship
        this->data.copy(ship.buffer, items_to_take, this->at_index);
        ship.structure = items_to_take;
        ship.has_payload = true;
        this->output_dock.ship_processed = true;

        // update our state
        this->at_index += items_to_take;

        // computation is done
        callback();
    }

private:
    std::string data;
    std::size_t chunk_size;
    std::size_t at_index;
};
