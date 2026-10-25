ifdef POLUS
ARCH = -mcpu=power8
endif
CXX = g++
CXXFLAGS = -O3 -std=c++11 -Wall -Wextra $(ARCH)

solver: solver.cpp
	$(CXX) $(CXXFLAGS) solver.cpp -o $@

clean:
	rm -f solver
