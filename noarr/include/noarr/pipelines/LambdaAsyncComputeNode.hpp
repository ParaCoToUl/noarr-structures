#ifndef NOARR_PIPELINES_LAMBDA_ASYNC_COMPUTE_NODE_HPP
#define NOARR_PIPELINES_LAMBDA_ASYNC_COMPUTE_NODE_HPP

#include <cstddef>
#include <vector>
#include <iostream>
#include <functional>

#include "noarr/pipelines/AsyncComputeNode.hpp"

namespace noarr {
namespace pipelines {

/**
 * A compute node built by passing in lambda expressions
 */
class LambdaAsyncComputeNode : public AsyncComputeNode {
private:
    std::function<void()> __impl__initialize;
    std::function<bool()> __impl__can_advance;
    std::function<void()> __impl__advance;
    std::function<void()> __impl__advance_async;
    std::function<void()> __impl__post_advance;
    std::function<void()> __impl__terminate;

public:
    LambdaAsyncComputeNode(const std::string& label) :
        AsyncComputeNode(label),
        __impl__initialize([](){}),
        __impl__can_advance([](){ return true; }),
        __impl__advance([&](){}),
        __impl__advance_async([&](){ this->callback(); }),
        __impl__post_advance([](){}),
        __impl__terminate([](){})
    { }

    LambdaAsyncComputeNode() :
        LambdaAsyncComputeNode(typeid(LambdaAsyncComputeNode).name())
    { };

public: // setting implementation
    void initialize(std::function<void()> impl) { __impl__initialize = impl; }
    void can_advance(std::function<bool()> impl) { __impl__can_advance = impl; }
    void advance(std::function<void()> impl) { __impl__advance = impl; }
    void advance_async(std::function<void()> impl) { __impl__advance_async = impl; }
    void post_advance(std::function<void()> impl) { __impl__post_advance = impl; }
    void terminate(std::function<void()> impl) { __impl__terminate = impl; }

protected: // using implementation
    void initialize() override { __impl__initialize(); }
    bool can_advance() override { return __impl__can_advance(); }
    void advance() override { return __impl__advance(); }
    void advance_async() override { return __impl__advance_async(); }
    void post_advance() override { return __impl__post_advance(); }
    void terminate() override { return __impl__terminate(); }  
};

} // pipelines namespace
} // namespace noarr

#endif
