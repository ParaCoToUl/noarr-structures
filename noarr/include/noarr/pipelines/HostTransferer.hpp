#ifndef NOARR_PIPELINES_HOST_TRANSFERER_HPP
#define NOARR_PIPELINES_HOST_TRANSFERER_HPP

#include <cstring>
#include <thread>
#include <functional>
#include <iostream>

#include "noarr/pipelines/MemoryTransferer.hpp"

namespace noarr {
namespace pipelines {

/**
 * Transfers within the host system
 * 
 * (not really useful for stuff other than debugging without a GPU)
 */
class HostTransferer : public MemoryTransferer {
private:
    bool use_background_thread;

public:
    HostTransferer(bool use_background_thread = false)
        : use_background_thread(use_background_thread)
    { }

    virtual void transfer(
        void* from,
        void* to,
        std::size_t bytes,
        std::function<void()> callback
    ) const override {
        if (use_background_thread) {
            std::thread t([from, to, bytes, callback](){
                std::memcpy(to, from, bytes);
                callback();
            });
            t.detach();
        } else {
            std::memcpy(to, from, bytes);
            callback();
        }
    }
};

} // pipelines namespace
} // namespace noarr

#endif
