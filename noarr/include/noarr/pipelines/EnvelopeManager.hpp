#ifndef NOARR_PIPELINES_ENVELOPE_MANAGER_HPP
#define NOARR_PIPELINES_ENVELOPE_MANAGER_HPP

#include <cstddef>
#include "noarr/pipelines/link.hpp"

namespace noarr {
namespace pipelines {

/**
 * TODO: implement this class
 */
template<typename Structure, typename BufferItem = void>
class EnvelopeManager {
private:
    std::size_t buffer_size;

public:
    EnvelopeManager(std::size_t buffer_size) {
        this->buffer_size = buffer_size;
    }
};

} // pipelines namespace
} // namespace noarr

#endif
