#ifndef NOARR_PIPELINES_CHUNK_STREAM_PROCESSOR_HPP
#define NOARR_PIPELINES_CHUNK_STREAM_PROCESSOR_HPP 1

namespace noarr {
namespace pipelines {

/**
 * Interface for objects that process chunk streams asynchronously
 * (all compute nodes and some envelopes)
 * 
 * Work happens on two levels and that dictates flags and the interface:
 * low level: processing of one chunk
 * high level: processing of the entire chunk stream
 */
class chunk_stream_processor {
public:
    /**
     * Returns true if this worker is ready for processing the next chunk
     */
    virtual bool is_ready_for_next_chunk() = 0;

    /**
     * Starts the worker on a next chunk
     * (override this, and call base)
     * 
     * TODO: replace this with promise API?
     */
    virtual void start_next_chunk_processing() {
        chunk_processing_finished = false;
    }

    /**
     * Returns true if the asnyc work on one chunk has finished
     */
    bool has_chunk_processing_finished() {
        return chunk_processing_finished;
    }

    /**
     * Returns true if this worker has processed all the chunks
     */
    bool has_processed_all_chunks() {
        return processed_all_chunks;
    }

protected:
    /**
     * Call this when one chunk processing finishes
     */
    bool set_chunk_processing_finished() {
        chunk_processing_finished = true;
    }

    /**
     * Call this when all chunks have been processed
     */
    bool set_all_chunks_processed() {
        processed_all_chunks = true;
    }

private:
    bool chunk_processing_finished = false;
    bool processed_all_chunks = false;
};

} // namespace pipelines
} // namespace noarr

#endif
