#ifndef NOARR_PIPELINES_MEMORY_TRANSFERER_HPP
#define NOARR_PIPELINES_MEMORY_TRANSFERER_HPP

#include <cassert>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

#include "noarr/pipelines/Device.hpp"

namespace noarr {
namespace pipelines {

/**
 * Transfers data between two devices
 */
class MemoryTransferer {
public:
    virtual void transfer(
        void* from,
        void* to,
        std::size_t bytes,
        std::function<void()> callback
    ) const = 0;
};

} // pipelines namespace
} // namespace noarr

#endif
