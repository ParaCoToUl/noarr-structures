#ifndef NOARR_STRUCT_DESC_HPP
#define NOARR_STRUCT_DESC_HPP

namespace noarr {

template<typename Name, typename Dims, typename ADims, typename... Params>
struct struct_desc;

template<typename>
struct type_param;

template<typename T, T V>
struct value_param;

}

#endif // NOARR_STRUCT_DESC_HPP