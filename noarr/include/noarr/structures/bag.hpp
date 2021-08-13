#ifndef NOARR_BAG_HPP
#define NOARR_BAG_HPP

#include <memory>

namespace noarr {

namespace helpers {

// general case for std::vector etc.
template<template<typename...> class container>
struct bag_policy {
	using type = container<char>;

	static auto construct(std::size_t size) {
		return container<char>(size);
	}

	static constexpr char* get(const container<char> &_container) {
		return _container.data();
	}
};

// a helper struct for 'bag_policy' as the '*' specifier is not a class ()
template<typename...>
struct bag_raw_pointer_tag;

template<>
struct bag_policy<std::unique_ptr> {
	using type = std::unique_ptr<char[]>;

	static auto construct(std::size_t size) {
		return std::make_unique<char[]>(size);
	}

	static constexpr char* get(const std::unique_ptr<char[]> &ptr) {
		return ptr.get();
	}
};

template<>
struct bag_policy<bag_raw_pointer_tag> {
	using type = char *;

	static char* construct(std::size_t size) {
		return new char[size];
	}

	static constexpr char* get(char *ptr) {
		return ptr;
	}

	static constexpr const char* get(const char *ptr) {
		return ptr;
	}
};

}

/**
 * @brief A bag is an abstraction of a structure combined with data of a corresponding size.
 * 
 * @tparam Structure: the structure that describes the data stored in the bag
 * @tparam BagPolicy: indicates what underlying structure contains the data blob (typically `std::unique_ptr`)
 */
template<typename Structure, typename BagPolicy>
struct bag {
private:
	typename BagPolicy::type data_;
	wrapper<Structure> structure_;

public:
	explicit bag(Structure s) : structure_(wrap(s)) { 
		data_ = BagPolicy::construct(structure().get_size());
	}

	explicit bag(wrapper<Structure> s) : structure_(s) {
		data_ = BagPolicy::construct(structure().get_size());
	}

	explicit bag(Structure s, typename BagPolicy::type &&data) : data_(std::move(data)), structure_(wrap(s))
	{ }

	explicit bag(wrapper<Structure> s, typename BagPolicy::type &&data) : data_(std::move(data)), structure_(s)
	{ }

	explicit bag(Structure s, typename BagPolicy::type &data) : data_(data), structure_(wrap(s))
	{ }

	explicit bag(wrapper<Structure> s, typename BagPolicy::type &data) : data_(data), structure_(s)
	{ }

	bag(Structure s, BagPolicy policy) : structure_(wrap(s)) { 
		data_ = policy.construct(structure().get_size());
	}

	bag(wrapper<Structure> s, BagPolicy policy) : structure_(s) {
		data_ = policy.construct(structure().get_size());
	}

	/**
	 * @brief return the wrapped structure which describes the `data` blob
	 */
	constexpr const wrapper<Structure>& structure() const noexcept { return structure_; }

	/**
	 * @brief returns the underlying data blob
	 */
	constexpr const char* data() const noexcept { return BagPolicy::get(data_); }

	/**
	 * @brief returns the underlying data blob
	 */
	constexpr char* data() noexcept { return BagPolicy::get(data_); }

	/**
	 * @brief sets the `data` to zeros
	 * 
	 */
	void clear() {
		auto size_ = structure().get_size();

		for (std::size_t i = 0; i < size_; ++i)
			data_[i] = 0;
	}

	/**
	 * @brief accesses a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @tparam Dims: the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, typename... Ts>
	constexpr decltype(auto) at(Ts... ts) const {
		return structure().template get_at<Dims...>(data(), ts...);
	}

	/**
	 * @brief accesses a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @tparam Dims: the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, typename... Ts>
	constexpr decltype(auto) at(Ts... ts) {
		return structure().template get_at<Dims...>(data(), ts...);
	}

	/**
	 * @brief returns an offset of a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @tparam Dims: the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, typename... Ts>
	constexpr auto offset(Ts... ts) const {
		return structure().template offset<Dims...>(ts...);
	}

	/**
	 * @brief returns an offset of a substructure with a certain index in a structure given by its dimension name
	 * 
	 * @tparam Dim: the dimension name
	 * @param t: the index of the substructure
	 */
	template<char Dim, typename T>
	constexpr auto get_offset(T t) const {
		return structure().template get_offset<Dim>(t);
	}

	/**
	 * @brief gets the length (number of indices) of a dimension in the `structure`
	 * 
	 * @tparam Dim: the dimension name
	 */
	template<char Dim>
	constexpr auto get_length() const {
		return structure().template get_length<Dim>();
	}

	/**
	 * @brief gets the size of the data described by the `structure`
	 * 
	 * @return constexpr auto 
	 */
	constexpr auto get_size() const {
		return structure().get_size();
	}
};

/**
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::unique_ptr
 * 
 * @param s: the structure
 */
template<typename Structure>
constexpr auto make_bag(Structure s) {
	return bag<Structure, helpers::bag_policy<std::unique_ptr>>(s);
}

/**
 * @brief creates a bag with the given structure and automatically creates the underlying data block implemented using std::unique_ptr
 * 
 * @param s: the structure (wrapped)
 */
template<typename Structure>
constexpr auto make_bag(noarr::wrapper<Structure> s) {
	return bag<Structure, helpers::bag_policy<std::unique_ptr>>(s);
}

/**
 * @brief creates a bag with the given structure and an underlying r/w observing data blob
 * 
 * @param s: the structure
 * @param data: the data blob
 */
template<typename Structure>
constexpr auto make_bag(Structure s, char *data) {
	return bag<Structure, helpers::bag_policy<helpers::bag_raw_pointer_tag>>(s, data);
}

/**
 * @brief creates a bag with the given structure and an underlying r/w observing data blob
 * 
 * @param s: the structure (wrapped)
 * @param data: the data blob
 */
template<typename Structure>
constexpr auto make_bag(noarr::wrapper<Structure> s, char *data) {
	return bag<Structure, helpers::bag_policy<helpers::bag_raw_pointer_tag>>(s, data);
}

} // namespace noarr

#endif // NOARR_BAG_HPP
