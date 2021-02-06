CXX=g++
STD=-std=c++14
OPTLEVEL=2
NVCCFLAGS=-ccbin $(CXX) -O$(OPTLEVEL) $(STD) -arch=sm_50 --expt-relaxed-constexpr
PROG=a.out

.PHONY: check clean all

all: $(PROG)

$(PROG): cuda.cu
	nvcc -o $@ $(NVCCFLAGS) -I. $< 

check: main.cpp
	g++  -o /dev/null -std=c++14 -Wall -Wextra -pedantic -Werror -I. $<
	clang++  -o /dev/null -std=c++14 -Wall -Wextra -pedantic -Werror -I. $<
	nvcc  -o /dev/null -std=c++14 -I. $<
	nvcc -o /dev/null $(NVCCFLAGS) -I. $<

clean:
	rm -f $(PROG)
