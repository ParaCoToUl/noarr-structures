#ifndef NOARR_STRUCT_DESC_HPP
#define NOARR_STRUCT_DESC_HPP

namespace noarr {

// TODO: add a way to get Params
// TODO: split type_param into struct_param and scalar_param

template<typename Name, typename Dims, typename ADims, typename... Params>
struct struct_description {
    using name = Name;
    using dims = Dims;
    using adims = ADims;
    using description = struct_description;
};

template<typename>
struct type_param;

template<typename T, T V>
struct value_param;

}

#endif // NOARR_STRUCT_DESC_HPP