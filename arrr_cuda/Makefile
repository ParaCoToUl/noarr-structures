CXX=g++
STD=-std=c++17
OPTLEVEL=2
CXXFLAGS=$(STD) -Wall -O$(OPTLEVEL) -g -ftemplate-backtrace-limit=0
NVCCFLAGS=-ccbin $(CXX) -O$(OPTLEVEL) $(STD) -arch=sm_50 --expt-relaxed-constexpr
LDFLAGS=
LIBS=
SRCS=$(wildcard *.cpp)
HDRS=$(wildcard *.hpp)
OBJS=$(SRCS:.cpp=.o)
PROG=arrr

$(PROG): cuda.cu arrr.hpp
	nvcc -o $@ $(NVCCFLAGS) $< 

clean:
	rm -f $(PROG) $(OBJS)
