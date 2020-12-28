OPTLEVEL=2
CXXFLAGS=-std=c++17 -Wall -O$(OPTLEVEL) -g -ftemplate-backtrace-limit=0
LDFLAGS=-g
LIBS=
CXX=g++
SRCS=$(wildcard *.cpp)
HDRS=$(wildcard *.hpp)
OBJS=$(SRCS:.cpp=.o)
PROG=arrr

$(PROG): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(PROG) $(OBJS)
