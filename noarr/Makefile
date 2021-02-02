CXX=g++
STD=-std=c++14
OPTLEVEL=2
NVCCFLAGS=-ccbin $(CXX) -O$(OPTLEVEL) $(STD) -arch=sm_50 --expt-relaxed-constexpr
PROG=a.out

$(PROG): cuda.cu
	nvcc -o $@ $(NVCCFLAGS) -I. $< 

clean:
	rm -f $(PROG)
