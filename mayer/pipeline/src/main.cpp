#include <iostream>
#include <vector>

class buffer {

};

class compute_node {
public:
    virtual void startExecute() = 0;
    virtual bool isDone() = 0;

    void addBuffer(buffer* b) {
        this->buffers.push_back(b);
    }

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
};

int main() {

    auto ld = my_loader();

    ld.startExecute();

    std::cout << "Example function!" << std::endl;
}
