#ifndef NOARR_PIPELINES_DEBUGGING_SCHEDULER_HPP
#define NOARR_PIPELINES_DEBUGGING_SCHEDULER_HPP

#include <iostream>
#include <string>
#include <exception>
#include <vector>

#include "Node.hpp"
#include "CompositeNode.hpp"
#include "SchedulerLogger.hpp"

namespace noarr {
namespace pipelines {

/*
    TODO: implement a proper "Scheduler" class that will do parallelism

    A naive scheduler implementation with no parallelism.

    TODO: parallelize memory transfers, device computation
        and host computation
 */

/**
 * A scheduler that executes nodes in a deterministic, synchronous way
 */
class DebuggingScheduler {
private:
    std::unique_ptr<SchedulerLogger> logger;

    /**
     * Nodes that the scheduler periodically updates
     */
    std::vector<Node*> nodes;

public:

    DebuggingScheduler() {
        //
    }

    DebuggingScheduler(std::ostream& output_stream) {
        this->logger = std::make_unique<SchedulerLogger>(output_stream);
    }

    ///////////////////////////////
    // Pipeline construction API //
    ///////////////////////////////

    /**
     * Registers a node to be updated by the scheduler
     */
    void add(Node& node) {
        this->nodes.push_back(&node);

        if (logger)
            logger->after_node_added(node);
    }

    /**
     * Registers a composite node to be updated by the scheduler
     */
    void add(CompositeNode& comp) {
        if (logger)
            logger->say("Adding composite node: " + comp.label + " ...");
        
        std::vector<Node*>& nodes = comp.scheduler_get_constituent_nodes();
        
        for (Node* n : nodes)
            this->add(*n);
    }

    ///////////////////
    // Execution API //
    ///////////////////

    /**
     * Runs the pipeline until no nodes can be advanced.
     */
    void run() {
        this->start_pipeline(); // need not be called explicitly

        while (!this->pipeline_finalized) {
            this->update_next_node();
            
            // NOTE: finalization is called automatically
        }
    }

    /////////////////////////////////////////////////////
    // Lower level API for more fine-grained debugging //
    /////////////////////////////////////////////////////

private:
    bool pipeline_started = false;
    bool pipeline_finalized = false;

    bool generation_advanced_data = false; // did this generation advance any data?
    std::size_t current_generation = 0;
    std::size_t next_node = 0;

public:
    /**
     * Scheduler looks at the next node and updates it
     * (it goes over the nodes in the order they were inserted,
     * generation by generation)
     * @return True if data was advanced
     */
    bool update_next_node() {
        if (this->nodes.size() == 0)
            throw std::runtime_error("Pipeline is empty so cannot be advanced");

        if (this->pipeline_finalized)
            throw std::runtime_error("Pipeline is already finalized");

        // pipeline starting
        if (!this->pipeline_started)
            this->start_pipeline();
        
        // === update the node ===
        
        Node* node = this->nodes[this->next_node];

        if (this->logger)
            this->logger->before_node_updated(*node);

        // update (synchronous)
        bool data_was_advanced;
        this->callback_will_be_called();
        node->scheduler_update([&](bool adv){
            data_was_advanced = adv;
            this->callback_was_called();
        });
        this->wait_for_callback(); // here's the synchronicity

        if (data_was_advanced)
            this->generation_advanced_data = true;

        // post update
        node->scheduler_post_update(data_was_advanced);

        // === send processed envelopes ===

        this->send_processed_envelopes(node);

        // === move to next node ===
        
        this->next_node++;

        // === move to next generation ===

        if (this->next_node >= this->nodes.size()) {
            if (this->logger) {
                this->logger->say(
                    "Generation " + std::to_string(this->current_generation)
                    + " has ended."
                );
            }

            // pipeline finalization
            if (!this->generation_advanced_data) {
                this->finalize_pipeline();
            }
            
            this->next_node = 0;
            this->current_generation++;
            this->generation_advanced_data = false;
        }

        return data_was_advanced;
    }

private:
    /**
     * Called automatically before the first node is updated
     */
    void start_pipeline() {
        if (this->pipeline_started)
            throw std::runtime_error("Pipeline was already started");

        for (Node* node : this->nodes)
            node->scheduler_start();

        this->pipeline_started = true;

        if (this->logger)
            this->logger->say("Pipeline started.");
    }

    /**
     * Called automatically after the pipeline detects stopping condition
     */
    void finalize_pipeline() {
        if (this->pipeline_finalized)
            throw std::runtime_error("Pipeline was already finalized");

        // TODO: implement any finalization logic here

        this->pipeline_finalized = true;

        if (this->logger)
            this->logger->say("Pipeline finalized.");
    }

    /**
     * Send all processed envelopes of a node
     */
    void send_processed_envelopes(Node* node) {
        for (UntypedPort* port : node->ports)
        {
            if (port->get_state() != PortState::processed)
                continue;

            if (port->envelope_target == nullptr)
                continue;

            if (port->envelope_target->get_state() != PortState::empty)
                continue;

            UntypedEnvelope* env = port->attached_envelope;
            port->attached_envelope = nullptr;
            port->envelope_target->attach_envelope(env);

            if (this->logger) {
                this->logger->after_envelope_sent(
                    *node,
                    *port->envelope_target->parent_node,
                    *env
                );
            }
        }
    }

    ///////////////////////////
    // Synchronization logic //
    ///////////////////////////

    // TODO: Add a proper synchronization primitive on which the scheduler
    // thread can wait and let the callback_was_called method (or its
    // quivalent) be callable from any thread.

private:
    bool _expecting_callback = false;
    bool _callback_was_called = false;

    /**
     * Call this before starting a node update
     */
    void callback_will_be_called() {
        if (this->_expecting_callback)
            throw std::runtime_error(
                "Cannot expect a callback when the previous didn't finish."
            );

        this->_expecting_callback = true;
        this->_callback_was_called = false;
    }

    /**
     * Call this from the node callback
     */
    void callback_was_called() {
        this->_callback_was_called = true;
    }

    /**
     * Call this to synchronously wait for the callback
     */
    void wait_for_callback() {
        if (!this->_expecting_callback)
            throw std::runtime_error(
                "Cannot wait for callback without first expecting it."
            );

        if (!this->_callback_was_called)
            throw std::runtime_error(
                "TODO: Asynchronous nodes are not implemented yet."
            );

        this->_expecting_callback = false;
    }

};

} // pipelines namespace
} // namespace noarr

#endif
