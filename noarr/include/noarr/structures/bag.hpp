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

	constexpr const noarr::wrapper<Structure>& structure() const noexcept { return structure_; }

	constexpr char* data() const noexcept { return data_.get(); }

	void clear() {
		auto size_ = structure().get_size();

		for (std::size_t i = 0; i < size_; ++i)
			data_[i] = 0;
	}

	template<char... Dims, typename... Ts>
	constexpr decltype(auto) at(Ts... ts) const {
		return structure().template get_at<Dims...>(data(), ts...);
	}

    template<char... Dims, typename... Ts>
    constexpr auto offset(Ts... ts) const {
        return structure().template offset<Dims...>(ts...);
    }

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