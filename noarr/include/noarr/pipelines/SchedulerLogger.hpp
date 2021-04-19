#ifndef NOARR_PIPELINES_SCHEDULER_LOGGER_HPP
#define NOARR_PIPELINES_SCHEDULER_LOGGER_HPP

#include <iostream>
#include <string>
#include <exception>

#include "Node.hpp"
#include "Envelope.hpp"

namespace noarr {
namespace pipelines {

/**
 * Performs logging for a scheduler with the goal of debugging a pipeline
 */
class SchedulerLogger {

    std::ostream& stream;
    std::string badge = "[scheduler]: ";

public:
    SchedulerLogger(std::ostream& stream) : stream(stream) { }

    void say(const std::string& message) {
        stream << badge << message << std::endl;
    }

    /**
     * Call this when a new node is added to the pipeline
     */
    void after_node_added(const Node& node) {
        say("Node has been added: " + node.label);
    }

    /**
     * Call this just before a node is updated
     */
    void before_node_updated(const Node& node) {
        say("Node will be updated: " + node.label);
    }

    /**
     * Call this after a processed envelope has been moved between nodes
     */
    void after_envelope_sent(
        const Node& from,
        const Node& to,
        const UntypedEnvelope& env
    ) {
        say(
            "Envelope is being sent: " +
            from.label + " --[" + env.label + "]--> " + to.label
        );
    }
};

} // pipelines namespace
} // namespace noarr

#endif
