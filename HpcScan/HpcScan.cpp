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
#include <fstream>

cl::Platform platform;
cl::Program program;
cl::Context context;
cl_int err = CL_SUCCESS;
std::vector<cl::Device> devices;

std::vector<int> sequential_exclusive_scan(std::vector<int> input);
std::vector<int> generate_vector(int min, int max, int size);


void print_vector(std::vector<int> vec);
void print_OpenCl_Error(cl::Error err);
bool setup_OpenCl_Platform();
void print_OpenCl_Platform(cl::Platform platform);

const std::string KERNEL_FILE = "scan.cl";
const std::string KERNEL = "scan_exclusive_sequential";
const int SELECTED_PLATFORM = 1;
const int BLOCK_SIZE = 128;
const int PROBLEM_SIZE = 20;
const int RANGE = 10;

const bool PRINT = true;

int main()
{
	auto vec = generate_vector(0, 10, PROBLEM_SIZE);
	print_vector(vec);
	auto vec_scanned = sequential_exclusive_scan(vec);
	std::cout << "Sequential C Result: " << std::endl;
	print_vector(vec_scanned);

	try {
		if (setup_OpenCl_Platform())
		{
			auto TIME_START = std::chrono::high_resolution_clock::now();
			cl::CommandQueue queue(context, devices[0], 0, &err);
			// INPUT BUFFER
			cl::Buffer source_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, PROBLEM_SIZE * sizeof(int));
			// OUTPUT BUFFER
			cl::Buffer dest_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, PROBLEM_SIZE * sizeof(int));

			// WRITE IMAGEDATA TO GPU BUFFER
			queue.enqueueWriteBuffer(
				source_buffer, // SOURCE BUFFER
				CL_TRUE, // BLOCK UNTIL OPERATION COMPLETE
				0, // OFFSET
				PROBLEM_SIZE * sizeof(int), // SIZE OF ARRAY 
				&vec[0] // POINTER TO VECTOR
			);

			// CREATE KERNEL
			cl::Kernel addKernel(program, KERNEL.c_str(), &err);

			// SET KERNEL ARGUMENTS
			addKernel.setArg(0, source_buffer); // PROBLEM ARRAY
			addKernel.setArg(1, dest_buffer); // SCANNED ARRAY
			addKernel.setArg(2, PROBLEM_SIZE);

			// SET ND RANGE FOR KERNEL (problem size, local size, global ranges, offset)
			cl::NDRange global(PROBLEM_SIZE);
			cl::NDRange local(1); //make sure local range is divisible by global range
			cl::NDRange offset(0); // START FROM 0,0

			std::cout << "CALL KERNEL " << KERNEL << std::endl;
			// ENQUEUE KERNEL EXECUTION
			queue.enqueueNDRangeKernel(addKernel, 0, global, local);

			// PREPARE AND WAIT FOR KERNEL RESULT
			cl::Event event;
			queue.enqueueReadBuffer(
				dest_buffer,
				CL_TRUE,
				0,
				vec.size() * sizeof(int),
				&vec_scanned[0],
				NULL,
				&event
			);

			// WAITING FOR KERNEL FINISH EVENT
			event.wait();
			auto TIME_END = std::chrono::high_resolution_clock::now();
			std::cout << "Received data from kernel" << std::endl;
			auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
			std::cout << "GPU Scan Execution: " << TOTAL_ELAPSED << " milliseconds";
			std::cout << "Scanned via GPU" << std::endl;
			if (PRINT)
				print_vector(vec_scanned);
		}
		else {
			return EXIT_FAILURE;
		}
	}
	catch (cl::Error err) {
		print_OpenCl_Error(err);
		return EXIT_FAILURE;
	}
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

bool setup_OpenCl_Platform()
{
	// get available platforms ( NVIDIA, Intel, AMD,...)
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cout << "No OpenCL platforms available!\n";
		return false;
	}
	platform = platforms[SELECTED_PLATFORM];
	print_OpenCl_Platform(platform);
	cl_context_properties properties[] =
	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

	devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// load and build the kernel
	std::ifstream sourceFile(KERNEL_FILE);
	if (!sourceFile)
	{
		std::cout << "kernel source file " << KERNEL_FILE << " not found!" << std::endl;
		return false;
	}
	std::string sourceCode(
		std::istreambuf_iterator<char>(sourceFile),
		(std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
	program = cl::Program(context, source);
	program.build(devices);
	return true;
}

void print_OpenCl_Platform(cl::Platform platform)
{
	// create a context and get available devices
	const cl_platform_info attributeTypes[5] = {
		CL_PLATFORM_NAME,
		CL_PLATFORM_VENDOR,
		CL_PLATFORM_VERSION,
		CL_PLATFORM_PROFILE,
		CL_PLATFORM_EXTENSIONS };

	std::cout << "Selected Platform Information: " << std::endl;
	std::cout << "Platform ID: " << SELECTED_PLATFORM << std::endl;
	std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "Platform Version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
	std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
}

void print_OpenCl_Error(cl::Error err)
{
	std::string s;
	program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &s);
	std::cout << s << std::endl;
	program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_OPTIONS, &s);
	std::cout << s << std::endl;

	std::cerr
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err.err()
		<< ")"
		<< std::endl;
}
