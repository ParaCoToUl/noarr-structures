#include <iostream>
#include <vector>
#include <string>
#include <tuple>

struct structure {
    
    // size of the data the structure represents (in bytes)
    std::size_t size();
    
    // fix a given dimension index
    template<char FixDim>
    void fix(std::size_t index);

    // return offset in bytes, relative to start of this structure,
    // given the current state of all indexes
    std::size_t offset();
};

template<typename TScalar>
struct scalar : structure {
    
    std::size_t size() {
        return sizeof(TScalar);
    }

    template<char FixDim>
    void fix(std::size_t index) {
        // do nothing
    }
    
    std::size_t offset() {
        return 0;
    }
};

template<char Dim, typename TSubstructure>
struct vector : structure {
    
    std::size_t size() {
        return length * substructure.size();
    }

    template<char FixDim>
    void fix(std::size_t index) {
        if (FixDim == Dim) {
            this->index = index;
        } else {
            // propagate to substructures
            substructure.template fix<FixDim>(index);
        }
    }

    std::size_t offset() {
        return index * substructure.size();
    }

private:
    std::size_t length;
    std::size_t index;
    TSubstructure substructure;
};

template<char Dim, typename... TSubstructures>
struct tuple : structure {

    std::size_t size() {
        return _compute_size<TSubstructures...>(substructures);
    }

    // helper recursive function
    template<typename TS, typename... TSs>
    static std::size_t _compute_size(std::tuple<TS, TSs...>& ts) {
        return std::get<0>(ts).size() + _compute_size(subtuple<1, TSs...>(ts));
    }

    // helper recursive function
    template<typename TS>
    static std::size_t _compute_size(std::tuple<TS>& ts) {
        return std::get<0>(ts).size();
    }

    // helper
    template<typename... T, std::size_t... I>
    static auto subtuple_(const std::tuple<T...>& t, std::index_sequence<I...>) {
        return std::make_tuple(std::get<I>(t)...);
    }

    // helper
    template<int Trim, typename... T>
    static auto subtuple(const std::tuple<T...>& t) {
        return subtuple_(t, std::make_index_sequence<sizeof...(T) - Trim>());
    }

    template<char FixDim>
    void fix(std::size_t index) {
        // TODO
    }

    std::size_t offset() {
        // TODO
    }

private:
    std::tuple<TSubstructures...> substructures;
};

// TODO: cube = vectorND, scube = arrayND
// ... remove vector and array as they clash with std:: variants
// TODO: what to rename "tuple" to, to not clash with std::tuple?

int main() {
    std::cout << "structure size: " << sizeof(structure) << std::endl;
    std::cout << "scalar<float> size: " << sizeof(scalar<float>) << std::endl;
    std::cout << "vector<scalar<float>> size: " << sizeof(vector<'i', scalar<float>>) << std::endl;

    // structure s = structure();
    // vector v = vector();
    // s.foo();
    // v.foo();

    auto foo = vector<'i', scalar<float>>();
    
    std::cout << "offset: " << foo.offset() << std::endl;
    
    foo.fix<'i'>(2);
    std::cout << "offset: " << foo.offset() << std::endl;

    // TUPLES!
    std::cout << "Tuples!" << std::endl;

    auto t = tuple<'t', scalar<float>, scalar<int>>();
    
    std::cout << "t.size(): " << t.size() << std::endl;
}
