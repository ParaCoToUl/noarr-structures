#ifndef NOARR_BAG_HPP
#define NOARR_BAG_HPP

namespace noarr {

/**
 * @brief A bag is an abstraction of a structure combined with data of a corresponding size.
 * 
 * @tparam Structure: the structure that describes the data stored in the bag
 */
template<typename Structure>
struct bag {
private:
	std::unique_ptr<char[]> data_;
	noarr::wrapper<Structure> structure_;

public:
	explicit bag(Structure s) : structure_(noarr::wrap(s)) { 
		data_ = std::make_unique<char[]>(structure().get_size());
	}

	explicit bag(noarr::wrapper<Structure> s) : structure_(s) {
		data_ = std::make_unique<char[]>(structure().get_size());
	}

	/**
	 * @brief return the wrapped structure which describes the `data` blob
	 */
	constexpr const noarr::wrapper<Structure>& structure() const noexcept { return structure_; }

	/**
	 * @brief returns the underlying data blob
	 */
	constexpr char* data() const noexcept { return data_.get(); }

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
	 * @brief returns an offset of a value in `data` by fixing dimensions in the `structure`
	 * 
	 * @tparam Dims : the dimension names
	 * @param ts: the dimension values
	 */
	template<char... Dims, typename... Ts>
	constexpr auto offset(Ts... ts) const {
		return structure().template offset<Dims...>(ts...);
	}

	/**
	 * @brief gets the length (number of indices) o a dimension in the `structure`
	 * 
	 * @tparam Dim: the dimension name
	 */
	template<char Dim>
	constexpr auto get_length() const {
		return structure().template get_length<Dim>();
	}

	constexpr auto get_size() const {
		return structure().template get_size();
	}
};

} // namespace noarr

#endif // NOARR_BAG_HPP