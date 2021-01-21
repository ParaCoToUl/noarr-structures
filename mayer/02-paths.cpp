#include <iostream>
#include <vector>
#include <string>

using namespace std;


/////////////////////////
// Forward Declaration //
/////////////////////////

struct path;

template<typename T>
struct scalar;

template<typename T>
struct p_scalar;

template<char label, size_t size, typename TItem>
struct array;

template<char label, size_t size, typename TItem>
struct p_array;


////////////////////
// Implementation //
////////////////////

struct path
{
    size_t size;
    std::vector<char> labels;
    std::vector<int> indices;

    static path empty()
    {
        return path(0);
    }

    path() : path(0) { }

    path(size_t size)
    {
        this->size = size;
        labels.resize(this->size);
        indices.resize(this->size);
    }

    path push(char label, int index)
    {
        path p = path(this->size + 1);
        
        for (int i = 0; i < this->size; i++)
        {
            p.labels.at(i) = this->labels.at(i);
            p.indices.at(i) = this->indices.at(i);
        }

        p.labels.at(this->size) = label;
        p.indices.at(this->size) = index;

        return p;
    }

    int find(char label)
    {
        for (int i = 0; i < this->size; i++)
        {
            if (this->labels.at(i) == label)
                return this->indices.at(i);
        }

        return -1;
    }
};

class IContainer
{
    public:
        virtual char* resolve_path(path p) = 0;
};

template<typename T>
struct scalar
{
    IContainer* tlc; // top level container
    path p;

    scalar(IContainer* tlc, path p)
    {
        this->tlc = tlc;
        this->p = p;
    }

    template<typename TContainer>
    static scalar<T> create_with_physical_layout()
    {
        return scalar<T>(
            new TContainer(),
            path::empty()
        );
    }

    T& ref()
    {
        char* ptr = tlc->resolve_path(p);

        return *((T*)ptr);
    }
};

template<typename T>
struct p_scalar : public IContainer
{
    T value;

    virtual char* resolve_path(path p)
    {
        return (char*) &value;
    }
};

template<char label, size_t size, typename P>
struct array
{
    IContainer* tlc; // top level container
    path p;

    array(IContainer* tlc, path p)
    {
        this->tlc = tlc;
        this->p = p;
    }

    template<typename TContainer>
    static array<label, size, P> create_with_physical_layout()
    {
        return array<label, size, P>(
            new TContainer(),
            path::empty()
        );
    }

    P at(int index)
    {
        return P(tlc, this->p.push(label, index));
    }
};

template<char label, size_t size, typename T>
struct p_array : public IContainer
{
    T items[size];

    T* ptr(int index)
    {
        return &items[index];
    }

    virtual char* resolve_path(path p)
    {
        int i = p.find(label);
        
        return this->items[i].resolve_path(p);
    }
};


/////////////
// Helpers //
/////////////

void print_memory(char* memory, int offset, int length)
{
    cout << "ADDR: ";
    for (int i = 0; i < length; i++)
    {
        printf("%4d ", offset + i); // address
    }
    cout << endl;

    cout << "DATA: ";
    for (int i = 0; i < length; i++)
    {
        cout << "-";
        printf("%02X", memory[offset + i] & 0xff);
        cout << "- ";
    }
    cout << endl;
}


//////////////////////////
// An Example Algorithm //
//////////////////////////

// virtual data layout
// (used to write the algorithm)
using channel = array<'x', 32, array<'y', 32, scalar<float>>>;
using image = array<'c', 3, channel>;

// physical data layout
// (used to actually store the data)
using p_image = p_array<'x', 32, p_array<'y', 32, p_array<'c', 3, p_scalar<float>>>>;

void load_image(image data)
{
    // populate the data container

    for (int c = 0; c < 3; c++)
    for (int x = 0; x < 32; x++)
    for (int y = 0; y < 32; y++)
        data.at(c).at(x).at(y).ref() = c + x + y;
}

float average_channel(channel c)
{
    float sum = 0.0f;

    for (int x = 0; x < 32; x++)
    for (int y = 0; y < 32; y++)
        sum += c.at(x).at(y).ref();

    return sum / (32.0f * 32.0f);
}

void run_algorithm()
{
    // allocate data
    image data = image::create_with_physical_layout<p_image>();
    
    //auto averages = array<'c', 3, scalar<float>>::create_with_implicit_layout(); // TODO
    auto averages = array<'c', 3, scalar<float>>::create_with_physical_layout<
        p_array<'c', 3, p_scalar<float>>
    >();

    // load the image
    load_image(data);

    // run the algorithm
    for (int c = 0; c < 3; c++)
        averages.at(c).ref() = average_channel(data.at(c));

    // print result
    printf("Averages: %f, %f, %f\n", averages.at(0).ref(), averages.at(1).ref(), averages.at(2).ref());

    // clean up stuff
    delete data.tlc;
    delete averages.tlc;
}


//////////
// Main //
//////////

int main()
{
    // THE ARCHITECTURE:
    // virtual datastructures -> path objects -> virtual data structures -> scalar values
    // physical data structures - embeded in virtual datastructures - can evaluate complete paths

    run_algorithm();

    // an empty line at the end
    cout << endl;
}
