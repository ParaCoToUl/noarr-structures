#ifndef NOARR_STRUCT_DESC_HPP
#define NOARR_STRUCT_DESC_HPP

#include "noarr_std_ext.hpp"

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

template<typename T, typename = void>
struct get_struct_desc;

template<typename T>
using get_struct_desc_t = typename get_struct_desc<T>::type;

// TODO: check if integral_pack
template<typename T>
struct get_struct_desc<T, void_t<typename T::description>> {
    using type = typename T::description;
};

}

#endif // NOARR_STRUCT_DESC_HPP