#ifndef NOARR_BAG_HPP
#define NOARR_BAG_HPP

#include <memory>
#include <vector>

#include "../extra/struct_traits.hpp"
#include "../extra/funcs.hpp"

namespace noarr {

namespace helpers {

// general case for std::vector etc.
template<template<class...> class container>
struct bag_policy {
	using type = container<char>;

	static constexpr auto construct(std::size_t size) {
		return container<char>(size);
	}

	static constexpr char *get(container<char> &_container) noexcept {
		return _container.data();
	}

	static constexpr const char *get(const container<char> &_container) noexcept {
		return _container.data();
	}
};

// a helper struct for 'bag_policy' as the '*' (pointer) specifier is not a class ()
template<class...>
struct bag_raw_pointer_tag;

// a helper struct for 'bag_policy' as the '*' (pointer) specifier is not a class ()
template<class...>
struct bag_const_raw_pointer_tag;

template<>
struct bag_policy<std::unique_ptr> {
	using type = std::unique_ptr<char[]>;

	static auto construct(std::size_t size) {
		return std::make_unique<char[]>(size);
	}

	static auto get(const std::unique_ptr<char[]> &ptr) noexcept {
		return ptr.get();
	}
};

template<>
struct bag_policy<bag_raw_pointer_tag> {
	using type = void *;

	template<class Ptr>
	static constexpr Ptr get(Ptr ptr) noexcept {
		return ptr;
	}
};

template<>
struct bag_policy<bag_const_raw_pointer_tag> {
	using type = const void *;

	static constexpr const void *get(const void *ptr) noexcept {
		return ptr;
	}
};

} // namespace helpers

/**
 * @brief A bag is an abstraction of a structure combined with data of a corresponding size.
 *
 * @tparam Structure: the structure that describes the data stored in the bag
 * @tparam BagPolicy: indicates what underlying structure contains the data blob (typically `std::unique_ptr`)
 */
template<class Structure, class BagPolicy>
class bag : contain<Structure> {
	using base = contain<Structure>;

	typename BagPolicy::type data_;

public:
	explicit constexpr bag(Structure s) : base(s), data_(BagPolicy::construct(s | noarr::get_size()))
	{ }

	explicit constexpr bag(Structure s, typename BagPolicy::type &&data) : base(s), data_(std::move(data))
	{ }

	explicit constexpr bag(Structure s, const typename BagPolicy::type &data) : base(s), data_(data)
	{ }

	explicit constexpr bag(Structure s, BagPolicy policy) : base(policy.construct(s)), data_(s | noarr::get_size())
	{ }

	/**
	 * @brief return the wrapped structure which describes the `data` blob
	 */
	constexpr auto structure() const noexcept { return base::template get<0>(); }

	/**
	 * @brief returns the underlying data blob
	 */
	constexpr auto data() const noexcept { return BagPolicy::get(data_); }

	/**
	 * @brief accesses a value in `data` by fixing dimensions in the `structure`
	 *
	 * @tparam Dims: the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, class... Ts>
	constexpr decltype(auto) at(Ts... ts) const noexcept {
		return structure() | noarr::get_at<Dims...>(data(), ts...);
	}

	/**
	 * @brief accesses a value in `data` by fixing dimensions in the `structure`
	 *
	 * @param ts: the dimension values
	 */
	template<class State>
	constexpr decltype(auto) operator[](State state) const noexcept {
		return structure() | noarr::get_at(data(), state);
	}

	/**
	 * @brief returns an offset of a value in `data` by fixing dimensions in the `structure`
	 *
	 * @tparam Dims: the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, class... Ts>
	constexpr auto offset(Ts... ts) const noexcept {
		return structure() | noarr::offset<Dims...>(ts...);
	}

	/**
	 * @brief gets the length (number of indices) of a dimension in the `structure`
	 *
	 * @tparam Dim: the dimension name
	 */
	template<char Dim>
	constexpr auto get_length() const noexcept {
		return structure() | noarr::get_length<Dim>();
	}

	/**
	 * @brief gets the size of the data described by the `structure`
	 *
	 */
	constexpr auto get_size() const noexcept {
		return structure() | noarr::get_size();
	}

	/**
	 * @brief wraps the data pointer in a new bag with the raw pointer policy. If the current bag owns the data,
	 * it continues owning them, while the returned raw bag should be considered a non-owning reference.
	 */
	constexpr auto get_ref() const noexcept;

	template<class ProtoStruct, class = std::enable_if_t<ProtoStruct::proto_preserves_layout>>
	friend constexpr auto operator ^(bag &&s, ProtoStruct p) {
		auto new_struct = s.structure() ^ p;
		return bag<decltype(new_struct), BagPolicy>(new_struct, std::move(s.data_));
	}


	template<class ProtoStruct, class = std::enable_if_t<
		ProtoStruct::proto_preserves_layout && std::is_trivially_copy_constructible_v<typename BagPolicy::type>>>
	friend constexpr auto operator ^(const bag &s, ProtoStruct p) {
		auto new_struct = s.structure() ^ p;
		return bag<decltype(new_struct), BagPolicy>(new_struct, s.data_);
	}

};

template<class Structure>
using unique_bag = bag<Structure, helpers::bag_policy<std::unique_ptr>>;

template<class Structure>
using raw_bag = bag<Structure, helpers::bag_policy<helpers::bag_raw_pointer_tag>>;

template<class Structure>
using const_raw_bag = bag<Structure, helpers::bag_policy<helpers::bag_const_raw_pointer_tag>>;

/**
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::unique_ptr
 *
 * @param s: the structure
 */
template<class Structure>
constexpr auto make_unique_bag(Structure s) noexcept {
	return unique_bag<Structure>(s);
}

/**
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::vector
 *
 * @param s: the structure
 */
template<class Structure>
constexpr auto make_vector_bag(Structure s) noexcept {
	return bag<Structure, helpers::bag_policy<std::vector>>(s);
}

/**
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::unique_ptr
 *
 * @param s: the structure
 */
template<class Structure>
constexpr auto make_bag(Structure s) noexcept {
	return make_unique_bag(s);
}

/**
 * @brief creates a bag with the given structure and an underlying r/w observing data blob
 *
 * @param s: the structure
 * @param data: the data blob
 */
template<class Structure>
constexpr auto make_bag(Structure s, void *data) noexcept {
	return raw_bag<Structure>(s, data);
}

/**
 * @brief creates a bag with the given structure and an underlying r/o observing data blob
 *
 * @param s: the structure
 * @param data: the data blob
 */
template<class Structure>
constexpr auto make_bag(Structure s, const void *data) noexcept {
	return const_raw_bag<Structure>(s, data);
}



template<class Structure, class BagPolicy>
constexpr auto bag<Structure, BagPolicy>::get_ref() const noexcept {
	return make_bag(structure(), data());
}



template<class T, class P>
struct to_struct<bag<T, P>> {
	using type = T;
	static constexpr T convert(const bag<T, P> &b) noexcept { return b.structure(); }
};

} // namespace noarr

#endif // NOARR_BAG_HPP
