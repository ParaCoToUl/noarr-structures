#include <cstddef>
#include <type_traits>
#include <iostream>

template<std::size_t S, typename T> struct array;
template<typename T = void> struct vector;
template<typename...Ts> struct tuple;

template<typename T>
struct traits {
    template<typename _T, typename = void>
    struct size_ {
        static constexpr std::size_t value = sizeof(_T);
    };

    template<typename...>
    using void_t = void;

    template<typename _T>
    struct size_<_T, void_t<decltype(_T::size)>> {
        static constexpr std::size_t value = _T::size;
    };
    
    static constexpr std::size_t size = size_<T>::value;
};

template<std::size_t S, typename T>
struct array {
private:
    template<typename I, typename... Is>
    struct at_type_ {
        using type = typename T::template at_type<Is...>;
    };

    template<typename I>
    struct at_type_<I> {
        using type = T*;
    };

public:
    static constexpr std::size_t size = S * traits<T>::size;

    template<typename I, typename... Is>
    using at_type = typename at_type_<I, Is...>::type;

    template<typename I, typename... Is>
    static at_type<I, Is...> at(void* data, I i, Is... is) {
        return T::at((void*)((char*)data + traits<T>::size * i), is...);
    }

    template<typename I>
    static at_type<I> at(void* data, I i) {
        return (at_type<I>)((char*)data + traits<T>::size * i);
    }
};

/*
 * tyhlety at_t, size, atd. by mely bejt ziskavany nakou traits classou jako u iteratoru treba
 * (protoze i primitivni float* je iterator, ale zadny float*::value_type neexistuje,
 * takze to vyrabi az ta traits classa co dostane float* jako parametr;
 * ale kdyz to je definovany v tom parametru, tak to proste veme tu definici z toho)
 * 
 * to co je v array (TODO: fill other implemented policies) private je jakoby substituce za takovej mechanismus
 */

int main() {
    char data[256];

    using policy1 = array<64, float>;
    using policy2 = array<8, array<8, float>>;

    volatile int x = 1;
    volatile int y = 0;

    *policy1::at(data, 0x8ULL) = 0x1P3;
    std::cout << *policy2::at(data, 1, 0) << std::endl;
    std::cout << *policy2::at(data, x, y) << std::endl;

    *policy1::at(data, 0x24ULL) = 0x9P2;
    std::cout << *policy2::at(data, 4, 4) << std::endl;

    *policy1::at(data, 0x3FULL) = 0x3FP0;
    std::cout << *policy2::at(data, 7, 7) << std::endl;
}