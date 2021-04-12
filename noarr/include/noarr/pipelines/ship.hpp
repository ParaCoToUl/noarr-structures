#ifndef NOARR_PIPELINES_SHIP_HPP
#define NOARR_PIPELINES_SHIP_HPP

#include <cstddef>
#include "untyped_ship.hpp"

namespace noarr {
namespace pipelines {

template<typename Structure, typename BufferItem = void>
class ship : public untyped_ship {
public:
    /**
     * The structure of data contained on the ship
     */
    Structure structure;

    /**
     * Pointer to the underlying data buffer
     */
    BufferItem* buffer;

    /**
     * Constructs a new ship from an existing buffer
     */
    ship(
        memory_device device,
        void* existing_buffer,
        std::size_t buffer_size
    ) : untyped_ship(device, existing_buffer, buffer_size) {
        this->buffer = (BufferItem*) existing_buffer;
    }

protected:
    // virtual method needed for polymorphism..
    // TODO: implement this class and add some virtual methods
    void foo() override {};
};

} // pipelines namespace
} // namespace noarr

#endif
