#include "stdafx.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <chrono>
#include <iostream>

std::vector<int> sequential_exclusive_scan(std::vector<int> input);
std::vector<int> generate_vector(int min, int max, int size);
void print_vector(std::vector<int> vec);

const int PROBLEM_SIZE = 20;
const int RANGE = 10;

int main()
{
	auto vec = generate_vector(0, 10, PROBLEM_SIZE);
	print_vector(vec);
	auto vec_scanned = sequential_exclusive_scan(vec);
	print_vector(vec_scanned);
    return EXIT_SUCCESS;
}

std::vector<int> sequential_exclusive_scan(std::vector<int> input)
{
	auto TIME_START = std::chrono::high_resolution_clock::now();
	std::vector<int> scanned = { 0 };
	for (unsigned int i = 1; i < input.size(); i++) {
		scanned.push_back(scanned[i - 1] + input.at(i - 1));
	}
	auto TIME_END = std::chrono::high_resolution_clock::now();
	auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
	std::cout << "Sequential Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
	return scanned;
}

void print_vector(std::vector<int> vec)
{
	int j = 1;
	for (unsigned int i = 0; i < vec.size(); i++)
	{
		std::cout << vec.at(i);
		if (i < vec.size()) {
			std::cout << ", ";
		}
		if (j > 15) {
			std::cout << std::endl;
			j = 0;
		}
		j++;
	}
	std::cout << std::endl;
}

std::vector<int> generate_vector(int min, int max, int size)
{
	std::vector<int> vec;
	for (int i = 0; i < size; i++)
	{
		vec.push_back(min + (rand() % static_cast<int>(max - min + 1)));
	}
	return vec;
}
