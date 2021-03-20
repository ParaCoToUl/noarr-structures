#include <iostream>
#include <vector>

class buffer {
    /*
        - needs an actual memory pointer for swapping buffers
        - needs to know gpu/cpu
        - needs to know when it's "empty"
        - needs to know when it contains EOS (other special messages?)

        Open issues:
        - strucutre? templates? ... or only blobs? blob dimensions?
     */
};

class compute_node {
public:
    virtual void startExecute() = 0;
    virtual bool isDone() = 0;

    // synchonously wait for completion
    virtual void waitForDone() = 0;

    void addBuffer(buffer* b);

protected:
    std::vector<buffer*> buffers;
};

class my_loader : public compute_node {
public:
    virtual void startExecute() {
        //
        //this->buffers.at(0)
    }

    virtual bool isDone() {
        return true;
    }

    virtual void waitForDone() {
        //
    }

    // can it produce one more chunk of input?
    virtual bool hasNext() {
        return true;
    }
};

// dummy inheritance just so I don't have to override abstract methods
class my_mapping_kernel : public my_loader { };
class my_aggregation_kernel : public my_loader { };
class my_printer : public my_loader { };

class linear_pipeline {
public:
    void start(const compute_node* n);
    void buffer(const buffer* n);
    
    void hostBuffer(std::size_t length);
    void deviceBuffer(std::size_t length);
    void copyNode();

    // executes the pipeline to completion
    void run();
};

int main() {

    /*
        OBSERVATIONS:
        - EOS ... end of chunk stream is needed to determine
            when to execute nodes after an aggregation node
    */


    // crap
    auto ml = new my_loader();
    auto b = new buffer()
    ml.setBuffer(b);


    // === build the pipeline ===

    auto pipeline = linear_pipeline();

    // load data from disk or somewhere
    pipeline.start(new my_loader());
    pipeline.hostBuffer(1024));
    
    // move the data to GPU
    pipeline.copyNode();
    pipeline.deviceBuffer(1024);

    // double-buffered
    // pipeline.bufferSwapNode();
    // pipeline.deviceBuffer(1024);

    // some kernel that maps chunks one-to-one
    pipeline.node(new my_mapping_kernel());
    pipeline.deviceBuffer(128);

    // double-buffered
    // pipeline.bufferSwapNode();
    // pipeline.deviceBuffer(128);

    // reduce node that aggregates the incomming chunk stream
    // and produces one chunk once it receives EOS from the input
    auto accumulator_buffer = buffer::create(..., 128); // pseudocode
    pipeline.node(new my_aggregation_kernel(accumulator_buffer));
    pipeline.deviceBuffer(128);

    // move the data to CPU
    pipeline.copyNode();
    pipeline.hostBuffer(128);

    // print the result
    pipeline.finish(new my_printer());

    // run the pipeline
    pipeline.run();

    /*
        HOW DOES THE (linear pipeline) SCHEDULER WORK?
        - each buffer knows, whether it's empty or filled with data
        - when a compute_node runs and no-longer needs a buffer,
            it marks it as empty
        - the scheduler runs nodes that
            a) have filled input buffers and have empty output buffers
            b) want to be run (see reduce nodes)
            -> both conditions are checked by the node itself,
                because it knows which buffers are input/output
     */
    
}
