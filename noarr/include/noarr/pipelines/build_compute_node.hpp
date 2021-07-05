#ifndef NOARR_PIPELINES_BUILD_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_BUILD_COMPUTE_NODE_HPP

#include <cstddef>
#include <vector>
#include <iostream>
#include <functional>

#include "noarr/pipelines/ComputeNode.hpp"

namespace noarr {
namespace pipelines {

// forward declaration
class ComputeNodeBuilder;

/**
 * A compute node built by passing implementation functions
 */
class BuiltComputeNode : public ComputeNode {
private:
    std::function<void()> __impl__initialize = nullptr;
    std::function<bool()> __impl__can_advance = nullptr;
    std::function<void()> __impl__advance = nullptr;
    std::function<void()> __impl__post_advance = nullptr;
    std::function<void()> __impl__terminate = nullptr;
    
    friend class ComputeNodeBuilder;

    // holds to builder instance to make sure its lifetime is the same as the
    // compute node and thus references to it won't die and so it can delegate
    // method calls like "link" or "callback"
    std::unique_ptr<ComputeNodeBuilder> builder;

public:
    BuiltComputeNode(std::function<void(ComputeNodeBuilder&)> factory);

    void initialize() override { __impl__initialize(); }
    bool can_advance() override { return __impl__can_advance(); }
    void advance() override { return __impl__advance(); }
    void post_advance() override { return __impl__post_advance(); }
    void terminate() override { return __impl__terminate(); }
};

/**
 * Delegates build calls to a BuiltComputeNode instance
 */
class ComputeNodeBuilder : public ComputeNode {
private:
    BuiltComputeNode& node;

public:
    ComputeNodeBuilder(BuiltComputeNode& node) : node(node) {}

    // method implementation
    void initialize(std::function<void()> impl) { node.__impl__initialize = impl; }
    void can_advance(std::function<bool()> impl) { node.__impl__can_advance = impl; }
    void advance(std::function<void()> impl) { node.__impl__advance = impl; }
    void post_advance(std::function<void()> impl) { node.__impl__post_advance = impl; }
    void terminate(std::function<void()> impl) { node.__impl__terminate = impl; }

    // regular method call redirection
    template<typename TLink> TLink& link(TLink& link) { return node.link(link); }
    void callback() { node.callback(); }
};

// constructor implementation
//  cannot be done by a factory method because we need the addres of "this"
//  to not change (as there will be a builder reference kept for long)
BuiltComputeNode::BuiltComputeNode(std::function<void(ComputeNodeBuilder&)> factory) {
    this->builder = std::make_unique<ComputeNodeBuilder>(*this);
    factory(*this->builder);
}

/**
 * Wraps around a compute node building factory
 */
BuiltComputeNode build_compute_node(std::function<void(ComputeNodeBuilder&)> factory) {
    // NOTE: Copy elision is needed here!
    // Because copying/moving the object breaks references to the builder inside.
    //
    // Luckily C++ 17 guarantees copy elision here and we do require that.
    // source: https://stackoverflow.com/a/12953129
    //
    // If you don't like this for some reason,
    // I suggest renaming BuiltComputeNode to build_compute_node
    return BuiltComputeNode(factory);
}

} // pipelines namespace
} // namespace noarr

#endif
