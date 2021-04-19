#include <string>
#include <functional>

#include <noarr/pipelines/PortState.hpp>
#include <noarr/pipelines/Port.hpp>
#include <noarr/pipelines/UntypedPort.hpp>
#include <noarr/pipelines/Node.hpp>

using namespace noarr::pipelines;

class MyConsumingNode : public Node {
public:
    Port<std::size_t, char> input_port;

    std::string received_string;

    MyConsumingNode() : Node(typeid(MyConsumingNode).name()) {
        this->received_string.clear();
    }

    virtual void register_ports(std::function<void(UntypedPort*)> register_port) {
        register_port(&this->input_port);
    };

    bool can_advance() override {
        // true, if we have a full ship available
        return this->input_port.get_state() == PortState::arrived;
    }

    void advance(std::function<void()> callback) override {
        // get the ship to be filled up
        auto& env = this->input_port.get_envelope();

        // move the chunk from ship into the accumulator
        this->received_string.append(env.buffer, env.structure);
        env.has_payload = false;
        this->input_port.envelope_processed = true;

        // computation is done
        callback();
    }
};
