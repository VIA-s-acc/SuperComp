ifdef POLUS
CXX = g++
ARCH = -mcpu=power8
else
CXX = /opt/homebrew/bin/g++-16
endif
CXXFLAGS = -O3 -std=c++11 -Wall -Wextra -fopenmp $(ARCH)

solver: solver.cpp
	$(CXX) $(CXXFLAGS) solver.cpp -o $@

clean:
	rm -f solver
