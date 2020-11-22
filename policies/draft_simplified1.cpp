#include <iostream>
#include <vector>

struct cross;

template<unsigned length> struct array;
template<typename ...Ts> struct shape;

template<typename T, typename... Ts, unsigned n>
struct shape<T, array<n>, cross, Ts...> {
private:
    using sub_shape = shape<T, Ts...>;

public:
    static constexpr unsigned size = sub_shape::size * n;

    template<typename index, typename ...indices>
    static T &get(T *memory, index first, indices... rest) {
        return sub_shape::get(memory + (sub_shape::size * first), rest...);
    }
};

template<typename T, unsigned n>
struct shape<T, array<n>> {
public:
    static constexpr unsigned size = n;

    template<typename index>
    static T &get(T *memory, index first) {
        return memory[first];
    }
};

int main() {
    using policy1 = shape<int, array<10>, cross, array<6>>;
    using policy2 = shape<int, array<5>, cross, array<12>>;
    using policy3 = shape<int, array<60>>;
    using policy4 = shape<int, array<2>, cross, array<5>, cross, array<6>>;

    std::cout << policy1::size << std::endl;
    std::cout << policy2::size << std::endl;
    std::cout << policy3::size << std::endl;
    std::cout << policy4::size << std::endl << std::endl;

    std::vector<int> vector(60);
    for (unsigned i = 0; i < 60; ++i)
        vector[i] = i;

    std::cout << policy1::get(vector.data(), 3, 2)    << std::endl;
    std::cout << policy2::get(vector.data(), 1, 8)    << std::endl;
    std::cout << policy3::get(vector.data(), 20)      << std::endl;
    std::cout << policy4::get(vector.data(), 0, 3, 2) << std::endl << std::endl;

    std::cout << policy1::get(vector.data(), 9, 5)    << std::endl;
    std::cout << policy2::get(vector.data(), 4, 11)    << std::endl;
    std::cout << policy3::get(vector.data(), 59)      << std::endl;
    std::cout << policy4::get(vector.data(), 1, 4, 5) << std::endl << std::endl;
}
