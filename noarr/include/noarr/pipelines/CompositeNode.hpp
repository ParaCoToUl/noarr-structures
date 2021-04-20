#ifndef NOARR_PIPELINES_COMPOSITE_NODE_HPP
#define NOARR_PIPELINES_COMPOSITE_NODE_HPP

#include <vector>
#include <string>
#include <cassert>

#include "Node.hpp"

namespace noarr {
namespace pipelines {

/**
 * A pipeline node that consists of multiple interconnected regular nodes,
 * it can be registered into a scheduler as a single component
 */
class CompositeNode {
public:
    std::string label;

    CompositeNode() : label(std::to_string((unsigned long)this)) { }

    CompositeNode(const std::string& label) : label(label) { }

private:
    bool registered_in_scheduler = false;
    std::vector<Node*> nodes;

protected:
    /**
     * Registers a node as a part of this composite node
     */
    void register_constituent_node(Node& node) {
        assert(!this->registered_in_scheduler
            && "Cannot add node when the composite node has "
            "been already registered in a scheduler");

        for (Node* n : this->nodes)
            assert(n != &node && "Cannot register a composite node twice");

        node.label = this->label + "::" + node.label;        
        this->nodes.push_back(&node);
    }

public:
    /**
     * Called by the scheduler during node registration,
     * it locks constituent node registration
     */
    std::vector<Node*>& scheduler_get_constituent_nodes() {
        this->registered_in_scheduler = true;
        return this->nodes;
    }
};

} // pipelines namespace
} // namespace noarr

#endif
