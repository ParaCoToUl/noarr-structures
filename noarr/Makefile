CXX=g++
STD=-std=c++14
OPTLEVEL=2
NVCCFLAGS=-ccbin $(CXX) -O$(OPTLEVEL) $(STD) -arch=sm_50 --expt-relaxed-constexpr
PROG=a.out

.PHONY: check check-gcc check-clang check-dpcpp check-nvcc1 check-nvcc2 clean all

all: $(PROG)

$(PROG): cuda.cu
	nvcc -o $@ $(NVCCFLAGS) -I. $< 

check: check-gcc check-clang check-dpcpp check-nvcc1 check-nvcc2
	@echo "everything OK"
check-gcc: main.cpp
	g++  -o /dev/null -std=c++14 -Wall -Wextra -pedantic -Werror -I. $< -pipe
check-clang: main.cpp
	clang++  -o /dev/null -std=c++14 -Wall -Wextra -pedantic -Werror -I. $< -pipe
check-dpcpp: main.cpp
	dpcpp  -o /dev/null -std=c++14 -Wall -Wextra -pedantic -Werror -I. $< -pipe
check-nvcc1: main.cpp
	nvcc -o /dev/null $(NVCCFLAGS) -I. $<
check-nvcc2: cuda.cu
	nvcc -o /dev/null $(NVCCFLAGS) -I. $<

clean:
	rm -f $(PROG)
