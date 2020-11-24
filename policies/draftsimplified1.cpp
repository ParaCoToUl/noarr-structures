template<unsigned length> struct array;
template<typename ...Ts> struct shape;
struct cross;

template<typename T, typename... Ts, unsigned n>
struct shape<T, array<n>, cross, Ts...> {
    using sub_shape = shape<T, Ts...>;
    static constexpr unsigned size = sub_shape::size * n;

    template<typename index, typename ...indices>
    static T &at(T *memory, index first, indices... rest) {
        return sub_shape::at(
            memory + (sub_shape::size * first), rest...);
    }
};

template<typename T, unsigned n>
struct shape<T, array<n>> {
    static constexpr unsigned size = n;

    template<typename index>
    static T &at(T *memory, index first) {
        return memory[first];
    }
};
