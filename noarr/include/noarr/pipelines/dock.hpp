#ifndef NOARR_PIPELINES_DOCK_HPP
#define NOARR_PIPELINES_DOCK_HPP

#include <cstddef>
#include "ship.hpp"
#include "untyped_dock.hpp"

namespace noarr {
namespace pipelines {

template<typename Structure, typename BufferItem = void>
class dock : public untyped_dock {
public:

    /**
     * Returns a reference to the docked ship
     */
    ship<Structure, BufferItem>& get_ship() {
        return dynamic_cast<ship<Structure, BufferItem>&>(
            this->get_untyped_ship()
        );
    }
};

} // pipelines namespace
} // namespace noarr

#endif
