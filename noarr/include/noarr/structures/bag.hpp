#ifndef NOARR_BAG_HPP
#define NOARR_BAG_HPP

namespace noarr {

template<typename Structure>
struct bag
{
private:
	std::unique_ptr<char[]> data_;
	noarr::wrapper<Structure> layout_;

public:
	explicit bag(Structure s)
		: layout_(noarr::wrap(s)),
		data_(std::make_unique<char[]>(layout().get_size())) { }
	explicit bag(noarr::wrapper<Structure> s)
		: layout_(s),
		data_(std::make_unique<char[]>(layout().get_size())) { }

	constexpr const noarr::wrapper<Structure>& layout() const noexcept { return layout_; }

	// noarr::wrapper<Structure> &layout() noexcept { return layout_; } // this version should reallocate the blob (maybe only if it doesn't fit)

	constexpr char* data() const noexcept { return data_.get(); }

	void clear()
	{
		auto size_ = layout().get_size();
		for (std::size_t i = 0; i < size_; i++)
			data_[i] = 0;
	}
};

}

#endif // NOARR_BAG_HPP