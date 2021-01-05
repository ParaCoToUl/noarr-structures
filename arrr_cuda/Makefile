CXX=g++
STD=-std=c++14
OPTLEVEL=2
NVCCFLAGS=-ccbin $(CXX) -O$(OPTLEVEL) $(STD) -arch=sm_50 --expt-relaxed-constexpr
PROG=arrr

$(PROG): cuda.cu arrr.hpp
	nvcc -o $@ $(NVCCFLAGS) $< 

clean:
	rm -f $(PROG)
