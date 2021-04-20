#include <string>
#include <functional>

#include <noarr/pipelines/PortState.hpp>
#include <noarr/pipelines/Port.hpp>
#include <noarr/pipelines/UntypedPort.hpp>
#include <noarr/pipelines/Node.hpp>

using namespace noarr::pipelines;

class MyProducingNode : public Node {
private:
    std::string data;
    std::size_t chunk_size;
    std::size_t at_index;

public:
    Port<std::size_t, char> output_port;

    MyProducingNode(const std::string& data, std::size_t chunk_size)
        : Node(typeid(MyProducingNode).name()),
        data(data), chunk_size(chunk_size), at_index(0),
        output_port(Device::HOST_INDEX) { }

    virtual void register_ports(std::function<void(UntypedPort*)> register_port) {
        register_port(&this->output_port);
    };

    bool can_advance() override {
        // false, if we've processed the entire dataset
        if (this->at_index >= this->data.length())
            return false;
        
        // true, if we have an empty envelope available
        return this->output_port.state() == PortState::arrived;
    }

    void advance(std::function<void()> callback) override {
        // get the envelope to be filled up
        auto& envelope = this->output_port.envelope();

        // compute the size of the next chunk
        std::size_t items_to_take = std::min(
            this->chunk_size,
            this->data.length() - this->at_index
        );

        // move the chunk onto the envelope
        this->data.copy(envelope.buffer, items_to_take, this->at_index);
        envelope.structure = items_to_take;
        envelope.has_payload = true;
        this->output_port.set_processed();

        // update our state
        this->at_index += items_to_take;

        // computation is done
        callback();
    }
};
