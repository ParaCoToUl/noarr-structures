#include <iostream>
#include <vector>
#include <string>

using namespace std;

/*
    Core idea: Data need not be serialized in memory - it can be a pointer structure
    otherwise it won't handle staggered data resizing well. We can ask it to serialize
    when being transferred to GPU or written to a file.

    At it's core, plain C++ exists. (arrays and vectors with regular sizeof).
    There's a wrapping layer that performs indexing in an abstract way that lets us swap out
    this underlying implementation. But the implementation is not slow - it's just as fast
    as regular C++ is!

    (not really regular C++ but buffed with some fancy methods that the wrapping layer needs)
    The wrapping layer = container class
*/

/////////////////////////
// Forward Declaration //
/////////////////////////

template<typename T>
struct scalar;

template<typename T>
struct scalar_slice;

template<char label, size_t size, typename TItem>
struct array;

template<char label, size_t size, typename TItem>
struct array_slice;

////////////////////
// Implementation //
////////////////////

template<typename T>
struct scalar
{
    T value;

    using slice = scalar_slice<T>;

    auto start_slice()
    {
        auto s = scalar_slice<T>();
        s.data = this;
        return s;
    }

    T* ptr()
    {
        return &value;
    }
};

template<typename T>
struct scalar_slice
{
    scalar<T>* data = NULL;

    template<char index_label>
    constexpr auto where(int index)
    {
        return *this;
    }

    auto ptr()
    {
        return data->ptr();
    }
};

template<char label, size_t size, typename TItem>
struct array
{
    TItem items[size];

    using slice = array_slice<label, size, TItem>;

    auto start_slice()
    {
        auto s = array_slice<label, size, TItem>();
        s.data = this;
        return s;
    }

    TItem* ptr(int index)
    {
        return &items[index];
    }
};

template<char label, size_t size, typename TItem>
struct array_slice
{
    array<label, size, TItem>* data = NULL;
    typename TItem::slice inner_slice;
    int index = -1;

    template<char index_label>
    constexpr auto where(int index)
    {
        if constexpr (label == index_label)
        {
            // consume
            this->index = index;
        }
        else
        {
            // pass on
            this->inner_slice.template where<index_label>(index);
        }
        
        return *this;
    }

    auto ptr()
    {
        // TODO: check all indices set properly

        this->inner_slice.data = data->ptr(this->index);
        
        return this->inner_slice.ptr();
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


//////////
// Main //
//////////

int main()
{
    // load data
    auto data = array<'x', 2, array<'y', 3, scalar<char>>>();
    auto data2 = array<'y', 3, array<'x', 2, scalar<char>>>();

    for (int x = 0; x < 2; x++)
    {
        for (int y = 0; y < 3; y++)
        {
            data.items[x].items[y].value = x * 0x10 + y;
            data2.items[y].items[x].value = x * 0x10 + y;
        }
    }

    // access data
    printf("10: %x\n", *data.start_slice().where<'x'>(1).where<'y'>(0).ptr());
    printf("10: %x\n", *data2.start_slice().where<'x'>(1).where<'y'>(0).ptr());

    // print the memory
    print_memory((char*) &data, 0, sizeof(data));
    print_memory((char*) &data2, 0, sizeof(data2));

    // an empty line at the end
    cout << endl;
}
