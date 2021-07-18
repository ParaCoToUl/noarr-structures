#ifndef NOARR_PIPELINES_LAMBDA_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_LAMBDA_COMPUTE_NODE_HPP

#include <cstddef>
#include <vector>
#include <iostream>
#include <functional>

#include "noarr/pipelines/ComputeNode.hpp"

namespace noarr {
namespace pipelines {

/**
 * A compute node built by passing in lambda expressions
 */
class LambdaComputeNode : public ComputeNode {
private:
    std::function<void()> __impl__initialize;
    std::function<bool()> __impl__can_advance;
    std::function<void()> __impl__advance;
    std::function<void()> __impl__post_advance;
    std::function<void()> __impl__terminate;

public:
    // constructor factory
    LambdaComputeNode(std::function<void(LambdaComputeNode&)> factory) :
        __impl__initialize([](){}),
        __impl__can_advance([](){ return false; }),
        __impl__advance([&](){ this->callback(); }),
        __impl__post_advance([](){}),
        __impl__terminate([](){})
    {
        factory(*this);
    }

    LambdaComputeNode() : LambdaComputeNode([](LambdaComputeNode& _){
        (void)_; // supress "unused variable" warning
    }) {};

public: // setting implementation
    void initialize(std::function<void()> impl) { __impl__initialize = impl; }
    void can_advance(std::function<bool()> impl) { __impl__can_advance = impl; }
    void advance(std::function<void()> impl) { __impl__advance = impl; }
    void post_advance(std::function<void()> impl) { __impl__post_advance = impl; }
    void terminate(std::function<void()> impl) { __impl__terminate = impl; }

protected: // using implementation
    void initialize() override { __impl__initialize(); }
    bool can_advance() override { return __impl__can_advance(); }
    void advance() override { return __impl__advance(); }
    void post_advance() override { return __impl__post_advance(); }
    void terminate() override { return __impl__terminate(); }  
};

} // pipelines namespace
} // namespace noarr

#endif
