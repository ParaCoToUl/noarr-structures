#ifndef NOARR_BAG_HPP
#define NOARR_BAG_HPP

#include <memory>
#include <vector>

#include "../extra/wrapper.hpp"

namespace noarr {

namespace helpers {

// general case for std::vector etc.
template<template<class...> class container>
struct bag_policy {
	using type = container<char>;

	static auto construct(std::size_t size) {
		return container<char>(size);
	}

	static char *get(container<char> &_container) noexcept {
		return _container.data();
	}

	static const char *get(const container<char> &_container) noexcept {
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

	static char *get(std::unique_ptr<char[]> &ptr) noexcept {
		return ptr.get();
	}

	static const char *get(const std::unique_ptr<char[]> &ptr) noexcept {
		return ptr.get();
	}
};

template<>
struct bag_policy<bag_raw_pointer_tag> {
	using type = char *;

	static char *construct(std::size_t size) {
		return new char[size];
	}

	static constexpr char *get(char *ptr) noexcept {
		return ptr;
	}

	static constexpr const char *get(const char *ptr) noexcept {
		return ptr;
	}
};

template<>
struct bag_policy<bag_const_raw_pointer_tag> {
	using type = const char *;

	static char *construct(std::size_t size) {
		return new char[size];
	}

	static constexpr const char *get(const char *ptr) noexcept {
		return ptr;
	}
};

template<class Structure, class BagPolicy>
class bag_base {
	template<class, class, class>
	friend class bag_impl;
private:
	typename BagPolicy::type data_;
	wrapper<Structure> structure_;

public:
	explicit constexpr bag_base(Structure s) : structure_(wrap(s)) {
		data_ = BagPolicy::construct(structure().get_size());
	}

	explicit constexpr bag_base(wrapper<Structure> s) : structure_(s) {
		data_ = BagPolicy::construct(structure().get_size());
	}

	explicit constexpr bag_base(Structure s, typename BagPolicy::type &&data) : data_(std::move(data)), structure_(wrap(s))
	{ }

	explicit constexpr bag_base(wrapper<Structure> s, typename BagPolicy::type &&data) : data_(std::move(data)), structure_(s)
	{ }

	explicit constexpr bag_base(Structure s, typename BagPolicy::type &data) : data_(data), structure_(wrap(s))
	{ }

	explicit constexpr bag_base(wrapper<Structure> s, typename BagPolicy::type &data) : data_(data), structure_(s)
	{ }

	constexpr bag_base(Structure s, BagPolicy policy) : structure_(wrap(s)) {
		data_ = policy.construct(structure().get_size());
	}

	constexpr bag_base(wrapper<Structure> s, BagPolicy policy) : structure_(s) {
		data_ = policy.construct(structure().get_size());
	}

	/**
	 * @brief return the wrapped structure which describes the `data` blob
	 */
	constexpr const wrapper<Structure>& structure() const noexcept { return structure_; }

	/**
	 * @brief returns the underlying data blob
	 */
	constexpr const char *data() const noexcept { return BagPolicy::get(data_); }

	/**
	 * @brief accesses a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @tparam Dims: the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, class... Ts>
	constexpr decltype(auto) at(Ts... ts) const noexcept {
		return structure().template get_at<Dims...>(data(), ts...);
	}

	/**
	 * @brief accesses a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @param ts: the dimension values
	 */
	template<class... Ts>
	constexpr decltype(auto) operator[](Ts... ts) const noexcept {
		return structure().template get_at(data(), ts...);
	}

	/**
	 * @brief returns an offset of a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @tparam Dims: the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, class... Ts>
	constexpr auto offset(Ts... ts) const noexcept {
		return structure().template offset<Dims...>(ts...);
	}

	/**
	 * @brief returns an offset of a substructure with a certain index in a structure given by its dimension name
	 * 
	 * @tparam Dim: the dimension name
	 * @param t: the index of the substructure
	 */
	template<char Dim, class T>
	constexpr auto get_offset(T t) const noexcept {
		return structure().template get_offset<Dim>(t);
	}

	/**
	 * @brief gets the length (number of indices) of a dimension in the `structure`
	 * 
	 * @tparam Dim: the dimension name
	 */
	template<char Dim, class... Ts>
	constexpr auto get_length(Ts... ts) const noexcept {
		return structure().template get_length<Dim>(ts...);
	}

	/**
	 * @brief gets the size of the data described by the `structure`
	 * 
	 */
	constexpr auto get_size() const noexcept {
		return structure().get_size();
	}

	/**
	 * @brief wraps the data pointer in a new bag with the raw pointer policy. If the current bag owns the data,
	 * it continues owning them, while the returned raw bag should be considered a non-owning reference.
	 */
	constexpr auto get_ref() const noexcept;
};

template<class Structure, class BagPolicy, class = void>
class bag_impl : public bag_base<Structure, BagPolicy> {
public:
	using bag_base<Structure, BagPolicy>::bag_base;
};

/**
 * @brief implementation for bag policies that define nonconst types
 */
template<class Structure, class BagPolicy>
class bag_impl<Structure, BagPolicy, std::enable_if_t<!std::is_const<std::remove_pointer_t<typename BagPolicy::type>>::value>>
	: public bag_base<Structure, BagPolicy> {
public:
	using bag_base<Structure, BagPolicy>::bag_base;
	using bag_base<Structure, BagPolicy>::structure;

	/**
	 * @brief returns the underlying data blob
	 */
	constexpr char *data() noexcept { return BagPolicy::get(bag_base<Structure, BagPolicy>::data_); }
	using bag_base<Structure, BagPolicy>::data;

	/**
	 * @brief sets the `data` to zeros
	 * 
	 */
	void clear() {
		auto size_ = structure().get_size();

		for (std::size_t i = 0; i < size_; ++i)
			bag_base<Structure, BagPolicy>::data_[i] = 0;
	}

	/**
	 * @brief accesses a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @tparam Dims: the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, class... Ts>
	constexpr decltype(auto) at(Ts... ts) noexcept {
		return structure().template get_at<Dims...>(data(), ts...);
	}
	using bag_base<Structure, BagPolicy>::at;

	/**
	 * @brief accesses a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @param ts: the dimension values
	 */
	template<class... Ts>
	constexpr decltype(auto) operator[](Ts... ts) noexcept {
		return structure().template get_at(data(), ts...);
	}
	using bag_base<Structure, BagPolicy>::operator[];

	/**
	 * @brief wraps the data pointer in a new bag with the raw pointer policy. If the current bag owns the data,
	 * it continues owning them, while the returned raw bag should be considered a non-owning reference.
	 */
	constexpr auto get_ref() noexcept;
	using bag_base<Structure, BagPolicy>::get_ref;
};

}

/**
 * @brief A bag is an abstraction of a structure combined with data of a corresponding size.
 * 
 * @tparam Structure: the structure that describes the data stored in the bag
 * @tparam BagPolicy: indicates what underlying structure contains the data blob (typically `std::unique_ptr`)
 */
template<class Structure, class BagPolicy>
class bag : public helpers::bag_impl<Structure, BagPolicy> {
public:
	using helpers::bag_impl<Structure, BagPolicy>::bag_impl;
};

/**
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::unique_ptr
 * 
 * @param s: the structure
 */
template<class Structure>
constexpr auto make_unique_bag(Structure s) noexcept {
	return bag<Structure, helpers::bag_policy<std::unique_ptr>>(s);
}

/**
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::unique_ptr
 * 
 * @param s: the structure (wrapped)
 */
template<class Structure>
constexpr auto make_unique_bag(noarr::wrapper<Structure> s) noexcept {
	return bag<Structure, helpers::bag_policy<std::unique_ptr>>(s);
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
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::vector
 * 
 * @param s: the structure (wrapped)
 */
template<class Structure>
constexpr auto make_vector_bag(noarr::wrapper<Structure> s) noexcept {
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
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::unique_ptr
 * 
 * @param s: the structure (wrapped)
 */
template<class Structure>
constexpr auto make_bag(noarr::wrapper<Structure> s) noexcept {
	return make_unique_bag(s);
}

/**
 * @brief creates a bag with the given structure and an underlying r/w observing data blob
 * 
 * @param s: the structure
 * @param data: the data blob
 */
template<class Structure>
constexpr auto make_bag(Structure s, char *data) noexcept {
	return bag<Structure, helpers::bag_policy<helpers::bag_raw_pointer_tag>>(s, data);
}

/**
 * @brief creates a bag with the given structure and an underlying r/w observing data blob
 * 
 * @param s: the structure (wrapped)
 * @param data: the data blob
 */
template<class Structure>
constexpr auto make_bag(noarr::wrapper<Structure> s, char *data) noexcept {
	return bag<Structure, helpers::bag_policy<helpers::bag_raw_pointer_tag>>(s, data);
}

/**
 * @brief creates a bag with the given structure and an underlying r/o observing data blob
 * 
 * @param s: the structure
 * @param data: the data blob
 */
template<class Structure>
constexpr auto make_bag(Structure s, const char *data) noexcept {
	return bag<Structure, helpers::bag_policy<helpers::bag_const_raw_pointer_tag>>(s, data);
}

/**
 * @brief creates a bag with the given structure and an underlying r/o observing data blob
 * 
 * @param s: the structure (wrapped)
 * @param data: the data blob
 */
template<class Structure>
constexpr auto make_bag(noarr::wrapper<Structure> s, const char *data) noexcept {
	return bag<Structure, helpers::bag_policy<helpers::bag_const_raw_pointer_tag>>(s, data);
}



template<class Structure, class BagPolicy>
constexpr auto helpers::bag_base<Structure, BagPolicy>::get_ref() const noexcept {
	return make_bag(structure(), data());
}

template<class Structure, class BagPolicy>
constexpr auto helpers::bag_impl<Structure, BagPolicy, std::enable_if_t<!std::is_const<std::remove_pointer_t<typename BagPolicy::type>>::value>>::get_ref() noexcept {
	return make_bag(structure(), data());
}



template<class T, class P>
struct to_struct<bag<T, P>> {
	using type = T;
	static constexpr T convert(const bag<T, P> &b) noexcept { return b.structure().unwrap(); }
};



template<class Struct, class BagPolicy, class ProtoStruct, class = std::enable_if_t<ProtoStruct::is_proto_struct>>
constexpr auto operator ^(bag<Struct, BagPolicy> &&s, ProtoStruct p) {
	auto new_struct = s.structure().unwrap() ^ p;
	return bag<decltype(new_struct), BagPolicy>(new_struct, std::move(s.data()));
}

} // namespace noarr

#endif // NOARR_BAG_HPP
