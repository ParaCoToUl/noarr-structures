CXX       :=  g++
STD       := -std=c++14
OPTLEVEL  := 2
CXXFLAGS  := -O$(OPTLEVEL) -Wall -Wextra -pedantic -Werror
INCLUDES  := ../noarr_pipelines/include
NVCCFLAGS := -ccbin $(CXX) -O$(OPTLEVEL) $(STD) -arch=sm_50 --expt-relaxed-constexpr
PROG      := a.out

INCLUDES_OPTS = $(patsubst %,-I%,$(INCLUDES))

.PHONY: check check-gcc check-clang check-icpx check-icpc check-nvcc1 check-nvcc2 clean all

all: $(PROG)

$(PROG): cuda.cu
	nvcc -o $@ $(NVCCFLAGS) $(INCLUDES_OPTS) $< 

check: check-gcc check-clang check-icpx check-icpc check-nvcc1 check-nvcc2
	@echo "everything OK"
check-gcc: main.cpp
	g++  -o /dev/null $(STD) $(CXXFLAGS) $(INCLUDES_OPTS) $< -pipe
check-clang: main.cpp
	clang++  -o /dev/null $(STD) $(CXXFLAGS) $(INCLUDES_OPTS) $< -pipe
check-icpx: main.cpp
	icpx  -o /dev/null $(STD) $(CXXFLAGS) $(INCLUDES_OPTS) $< -pipe
check-icpc: main.cpp
	icpc  -o /dev/null $(STD) $(CXXFLAGS) $(INCLUDES_OPTS) $< -pipe
check-nvcc1: main.cpp
	nvcc -o /dev/null $(NVCCFLAGS) $(INCLUDES_OPTS) $<
check-nvcc2: cuda.cu
	nvcc -o /dev/null $(NVCCFLAGS) $(INCLUDES_OPTS) $<

clean:
	rm -f $(PROG)
