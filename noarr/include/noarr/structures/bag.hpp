#ifndef NOARR_BAG_HPP
#define NOARR_BAG_HPP

namespace noarr {

/**
 * @brief A bag is an abstraction of a structure combined with data of a corresponding size.
 * 
 * @tparam Structure - the structure that describes the data stored in the bag
 */
template<typename Structure>
struct bag
{
private:
	std::unique_ptr<char[]> data_;
	noarr::wrapper<Structure> structure_;

public:
	explicit bag(Structure s)
		: structure_(noarr::wrap(s))
		{ 
			data_ = std::make_unique<char[]>(structure().get_size());
		}

	explicit bag(noarr::wrapper<Structure> s)
		: structure_(s)
		{
			data_ = std::make_unique<char[]>(structure().get_size());
		}

	constexpr const noarr::wrapper<Structure>& structure() const noexcept { return structure_; }

	constexpr char* data() const noexcept { return data_.get(); }

	void clear()
	{
		auto size_ = structure().get_size();

		for (std::size_t i = 0; i < size_; i++)
			data_[i] = 0;
	}
};

}

#endif // NOARR_BAG_HPP