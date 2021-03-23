#ifndef NOARR_PIPELINES_LINEAR_PIPELINE_HPP
#define NOARR_PIPELINES_LINEAR_PIPELINE_HPP 1

#include <memory>
#include <vector>
#include "noarr/pipelines/chunk_stream_processor.hpp"
#include "noarr/pipelines/envelope.hpp"
#include "noarr/pipelines/compute_node.hpp"
#include "noarr/pipelines/producer_compute_node.hpp"
#include "noarr/pipelines/consumer_compute_node.hpp"
#include "noarr/pipelines/pipe_compute_node.hpp"

namespace noarr {
namespace pipelines {

/**
 * Simplifies the process of building out a linear pipeline
 */
class linear_pipeline {
public:

    ///////////////////
    // Builder logic //
    ///////////////////

    class node_builder {
        void node() {
            //
        }
    };

    template<typename Structure, typename BufferItem = void>
    class envelope_builder {
    private:
        std::unique_ptr<linear_pipeline> pipeline;

    public:        
        envelope_builder(std::unique_ptr<linear_pipeline>&& pipeline) {
            this->pipeline = std::move(pipeline);
        }

        linear_pipeline envelope_existing(
            pipe_envelope<Structure, BufferItem>* env
        ) {
            this->pipeline->all_processors.push_back(env);
            return this->pipeline;
        }

        // template<typename PipeEnvelope>
        // linear_pipeline envelope_existing(PipeEnvelope&& env) {
        //     static_assert(
        //         std::is_base_of<pipe_envelope, PipeEnvelope>::value,
        //         "given envelope instance must derive from 'pipe_envelope'"
        //     );
        //     this->pipeline->chunk_stream_processors.push_back(&env);
        //     return this->pipeline;
        // }
    };

    class initial_builder {
    private:
        std::unique_ptr<linear_pipeline> pipeline;
    
    public:
        initial_builder(std::unique_ptr<linear_pipeline>&& pipeline) {
            this->pipeline = std::move(pipeline);
        }

        // template<typename ProducerNode, typename... Args>
        // linear_pipeline start_node(Args&&... args) {
        //     auto node = std::make_unique<ProducerNode>(
        //         std::forward<Args>(args)...
        //     );
        //     // TODO
        // }

        template<class ProducerComputeNode>
        using get_structure = void; // .............. ????

        template<
            class ProducerComputeNode,
            template<typename _S, typename _BI> class PCNParent,
            typename Structure,
            typename BufferItem
        >
        void foo() {
            static_assert(
                std::is_base_of<
                    producer_compute_node<
                        Structure,
                        BufferItem
                    >,
                    ProducerComputeNode
                >::value,
                "given node must derive from 'producer_compute_node'"
            );
        }

        /**
         * Creates new start node of the pipeline, of the given type
         * and constructor 
         * 
         * TODO: doesn't work for some reason?
         */
        template<
            class ProducerComputeNode,
            typename Structure,
            typename BufferItem,
            typename... Args
        >
        envelope_builder<Structure, BufferItem> start_node(Args&&... args) {
            static_assert(
                std::is_base_of<
                    producer_compute_node<Structure, BufferItem>,
                    ProducerComputeNode
                >::value,
                "given node must derive from 'producer_compute_node'"
            );
            auto node = static_cast<
                std::unique_ptr<producer_compute_node<Structure, BufferItem>>
            >(
                std::make_unique<ProducerComputeNode>(
                    std::forward<Args>(args)...
                )
            );
            this->pipeline->all_processors.push_back(&*node);
            this->pipeline->owned_processors.push_back(std::move(node));
            return envelope_builder(this->pipeline);
        }

        /**
         * Creates a new start node of the pipeline, by providing a fresh
         * new node instance to move
         */
        template<
            template<typename _Structure, typename _BufferItem>
                class ProducerComputeNode,
            typename Structure,
            typename BufferItem
        >
        envelope_builder<Structure, BufferItem> start_node_new(
            std::unique_ptr<ProducerComputeNode<Structure, BufferItem>>&& node
        ) {
            static_assert(
                std::is_base_of<
                    producer_compute_node<Structure, BufferItem>,
                    ProducerComputeNode<Structure, BufferItem>
                >::value,
                "given node must derive from 'producer_compute_node'"
            );
            this->pipeline->all_processors.push_back(&*node);
            this->pipeline->owned_processors.push_back(std::move(node));
            return envelope_builder(this->pipeline);
        }

        /**
         * Specifies the start node of the pipeline as an external,
         * already existing node, to which a pointer is given
         */
        template<
            template<typename _Structure, typename _BufferItem>
                class ProducerComputeNode,
            typename Structure,
            typename BufferItem
        >
        envelope_builder<Structure, BufferItem> start_node_existing(
            ProducerComputeNode<Structure, BufferItem>* node_ptr
        ) {
            static_assert(
                std::is_base_of<
                    producer_compute_node<Structure, BufferItem>,
                    ProducerComputeNode<Structure, BufferItem>
                >::value,
                "given node must derive from 'producer_compute_node'"
            );
            this->pipeline->all_processors.push_back(node_ptr);
            return envelope_builder(this->pipeline);
        }
    };

    /**
     * Initiates linear pipeline building
     */
    static initial_builder builder() {
        return initial_builder(
            std::make_unique<linear_pipeline>()
        );
    }

    ////////////////////
    // Pipeline logic //
    ////////////////////

    std::vector<std::unique_ptr<chunk_stream_processor>> owned_processors;
    std::vector<chunk_stream_processor*> all_processors;
    chunk_stream_processor* end_node;

    /**
     * Run the linear pipeline to completion
     */
    void run() {
        // TODO: verify the pipeline has been fully built

        // execute workers until the final compute node
        // has received end of stream
        // TODO ...

        while (!this->end_node->has_processed_all_chunks())
        {
            for (chunk_stream_processor* processor : all_processors)
            {
                if (processor->is_ready_for_next_chunk())
                    processor->start_next_chunk_processing();
            }

            // thread.yield();
        }
    }
};

} // namespace pipelines
} // namespace noarr

#endif
