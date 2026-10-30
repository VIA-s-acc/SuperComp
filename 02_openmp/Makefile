ifdef POLUS
CXX = g++
ARCH = -mcpu=power8
else
CXX = /opt/homebrew/bin/g++-16
endif
CXXFLAGS = -O3 -std=c++11 -Wall -Wextra -fopenmp $(ARCH)

.PHONY: test clean

solver: solver.cpp solver.h problem.h
	$(CXX) $(CXXFLAGS) solver.cpp -o $@

test: run_tests
	./run_tests

run_tests: test.cpp solver.h problem.h
	$(CXX) $(CXXFLAGS) test.cpp -o $@

clean:
	rm -f solver run_tests
