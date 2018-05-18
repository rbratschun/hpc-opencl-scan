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

#pragma region global
cl::Platform platform;
cl::Program program;
cl::Context context;
cl_int err = CL_SUCCESS;
std::vector<cl::Device> devices;
#pragma endregion

#pragma region vector_methods
std::vector<int> generate_vector(int min, int max, int size);
std::vector<int> sequential_exclusive_scan(std::vector<int> input);
std::vector<int> opencl_WE_prefixsum(std::vector<int> input, std::vector<int> &blockSums, int BLOCK_SIZE);
std::vector<int> opencl_scan_large_array(std::vector<int> input);
void opencl_WE_scanBlockSums(std::vector<int> &blockSums, std::vector<int> &blockSumsScanned);
void opencl_WE_addBlockSums(std::vector<int> &output, std::vector<int> blockSums);
#pragma endregion

#pragma region helpers
void print_vector(std::vector<int> vec, std::string info);
void print_OpenCl_Error(cl::Error err);
bool setup_OpenCl_Platform();
void print_OpenCl_Platform(cl::Platform platform);
std::vector<int> opencl_naive_scan(std::vector<int> vec);
#pragma endregion

#pragma region CONSTANTS
// local work_size
const int BLOCK_SIZE = 32; // = 32 * 2
// PROBLEM_SIZE = vielfaches von BLOCK_SIZE - muss quadrat von 2 entsprechen
// ACHTUNG - derzeit noch begrenzt auf ~ single thread kapazität -> ~1024
const int PROBLEM_SIZE = BLOCK_SIZE * 36384 * 2 * 16;
// maximum wert integer
const int RANGE = 10;
// Ausgabe der Vektoren
const bool PRINT = false;
const bool DEBUG = false;
// OpenCl Kernel script & 
const std::string KERNEL_FILE = "scan.cl";
#pragma endregion

int main()
{
	// TODO: setup console parameters for PROBLEM_SIZE & PRINT
	// alternatively std::cin menu
	auto vec = generate_vector(0, 10, PROBLEM_SIZE);
	auto  vec_scan_seq = sequential_exclusive_scan(vec);
	std::vector<int> vec_gpu;
	if (setup_OpenCl_Platform()) {
		vec_gpu = opencl_scan_large_array(vec);
		auto result = vec_gpu[PROBLEM_SIZE - 1] == vec_scan_seq[PROBLEM_SIZE - 1] ? "true" : "false";
		std::cout << "last item of vectors equal ? " << result << std::endl;;
		std::cout << "sequential: " << vec_scan_seq[PROBLEM_SIZE - 1] << ", gpu: " << vec_gpu[PROBLEM_SIZE - 1] << std::endl;
	}
	else
		return EXIT_FAILURE;

    return EXIT_SUCCESS;
}


std::vector<int> opencl_scan_large_array(std::vector<int> input) {

	std::vector<int> blockSums(input.size() / BLOCK_SIZE);
	std::vector<int> blockSumsScanned(input.size() / BLOCK_SIZE);

	auto TIME_START = std::chrono::high_resolution_clock::now();
	/* Phase 1: Scan single Blocks of Problem */
	auto output = opencl_WE_prefixsum(input, blockSums, BLOCK_SIZE);
	/* Phase 2: Scan Blocksums vector */
	opencl_WE_scanBlockSums(blockSums, blockSumsScanned);
	/* Phase 3: Add BlocksumsScanned to Output */
	opencl_WE_addBlockSums(output, blockSumsScanned);
	/* Finished ! */
	auto TIME_END = std::chrono::high_resolution_clock::now();
	auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
	std::cout << "" << TOTAL_ELAPSED << " milliseconds" << std::endl;
	std::cout << "**********************************************" << std::endl;
	return output;
}

std::vector<int> opencl_WE_prefixsum(std::vector<int> input, std::vector<int> &blockSums, int BLOCK_SIZE)
{
	std::vector<int> output(input.size());

	try {
		cl::CommandQueue queue(context, devices[0], 0, &err);
		auto TIME_START = std::chrono::high_resolution_clock::now();
		// INPUT BUFFER
		cl::Buffer input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, PROBLEM_SIZE * sizeof(int));
		// OUTPUT BUFFER
		cl::Buffer output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, PROBLEM_SIZE * sizeof(int));

		// INPUT BUFFER
		cl::Buffer blocksums_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, blockSums.size() * sizeof(int));

		queue.enqueueWriteBuffer(
			input_buffer, // SOURCE BUFFER
			CL_TRUE, // BLOCK UNTIL OPERATION COMPLETE
			0, // OFFSET
			PROBLEM_SIZE * sizeof(int), // SIZE OF ARRAY 
			&input[0] // POINTER TO VECTOR
		);

		// CREATE KERNEL
		const std::string KERNEL = "prefixsum";
		cl::Kernel addKernel(program, KERNEL.c_str(), &err);

		// SET KERNEL ARGUMENTS
		addKernel.setArg(0, input_buffer); // PROBLEM ARRAY
		addKernel.setArg(1, output_buffer); // SCANNED ARRAY
		int LOCAL_SIZE = sizeof(cl_int) * BLOCK_SIZE;
		addKernel.setArg(2, cl::LocalSpaceArg(cl::Local(LOCAL_SIZE))); // BLOCK ARRAY
		addKernel.setArg(3, blocksums_buffer); // SCANNED ARRAY
		addKernel.setArg(4, BLOCK_SIZE); // PROBLEM SIZE

		// SET ND RANGE FOR KERNEL (problem size, local size, global ranges, offset)
		// const int GLOBAL_WORK_SIZE = (input.size() / (input.size() / WORK_GROUP_SIZE));
		const int GLOBAL_WORK_SIZE = input.size();
		const int LOCAL_WORK_SIZE = BLOCK_SIZE;
		cl::NDRange global(GLOBAL_WORK_SIZE);
		cl::NDRange local(LOCAL_WORK_SIZE); //make sure local range is divisible by global range
		// cl::NDRange offset(0); // START FROM 0,0

		std::cout << "CALL KERNEL " << KERNEL << std::endl;
		// ENQUEUE KERNEL EXECUTION
		// queue.enqueueNDRangeKernel(addKernel, 0, global, local);
		queue.enqueueNDRangeKernel(addKernel, 1, global, local);

		queue.finish();
		// PREPARE AND WAIT FOR KERNEL RESULT
		cl::Event event;
		queue.enqueueReadBuffer(
			output_buffer,
			CL_TRUE,
			0,
			output.size() * sizeof(int),
			&output[0],
			NULL,
			&event
		);

		queue.enqueueReadBuffer(
			blocksums_buffer,
			CL_TRUE,
			0,
			blockSums.size() * sizeof(int),
			&blockSums[0],
			NULL,
			&event
		);

		// WAITING FOR KERNEL FINISH EVENT
		event.wait();
		auto TIME_END = std::chrono::high_resolution_clock::now();
		std::cout << "Received data from kernel" << std::endl;
		auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
		std::cout << "GPU Scan Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
		std::cout << "**********************************************" << std::endl;
		if (DEBUG) {
			print_vector(output, "Output prefixsum ");
			print_vector(blockSums, "blockSums");
		}
	}
	catch (cl::Error err) {
		print_OpenCl_Error(err);
	}
	return output;
}

void opencl_WE_scanBlockSums(std::vector<int> &blockSums, std::vector<int> &blockSumsScanned)
{
	std::cout << "Scanning blocksums ... " << std::endl;
	std::vector<int> blockSumsBuffer(blockSums.size());
	cl::CommandQueue queue(context, devices[0], 0, &err);
	auto TIME_START = std::chrono::high_resolution_clock::now();
	// INPUT BUFFER
	cl::Buffer input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, blockSums.size() * sizeof(int));
	// OUTPUT BLOCKSUMS BUFFER
	cl::Buffer output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, blockSums.size() * sizeof(int));
	// LOCAL BLOCKSUMS BUFFER
	cl::Buffer blocksums_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, blockSums.size() * sizeof(int));

	queue.enqueueWriteBuffer(
		input_buffer, // SOURCE BUFFER
		CL_TRUE, // BLOCK UNTIL OPERATION COMPLETE
		0, // OFFSET
		blockSums.size() * sizeof(int), // SIZE OF ARRAY 
		&blockSums[0] // POINTER TO VECTOR
	);

	// CREATE KERNEL
	const std::string KERNEL = "scan";
	cl::Kernel addKernel(program, KERNEL.c_str(), &err);

	// SET KERNEL ARGUMENTS
	addKernel.setArg(0, input_buffer); // PROBLEM ARRAY
	addKernel.setArg(1, output_buffer); // SCANNED ARRAY
	addKernel.setArg(2, blockSums.size()); // PROBLEM SIZE

									   // SET ND RANGE FOR KERNEL (problem size, local size, global ranges, offset)
	const int GLOBAL_WORK_SIZE = blockSums.size();
	cl::NDRange global(GLOBAL_WORK_SIZE);

	std::cout << "CALL KERNEL " << KERNEL << std::endl;
	// ENQUEUE KERNEL EXECUTION
	queue.enqueueNDRangeKernel(addKernel, 0, global);

	// PREPARE AND WAIT FOR KERNEL RESULT
	cl::Event event;
	queue.enqueueReadBuffer(
		output_buffer,
		CL_TRUE,
		0,
		blockSumsScanned.size() * sizeof(int),
		&blockSumsScanned[0],
		NULL,
		&event
	);

	queue.enqueueReadBuffer(
		blocksums_buffer,
		CL_TRUE,
		0,
		blockSumsBuffer.size() * sizeof(int),
		&blockSumsBuffer[0],
		NULL,
		&event
	);

	// WAITING FOR KERNEL FINISH EVENT
	event.wait();
	auto TIME_END = std::chrono::high_resolution_clock::now();
	std::cout << "Received data from kernel" << std::endl;
	auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
	std::cout << "Scan BlockSums elapsed in " << TOTAL_ELAPSED << " milliseconds" << std::endl;
	std::cout << "**********************************************" << std::endl;
	if (DEBUG) {
		print_vector(blockSumsScanned, "Output blockSumsScanned ");
		// print_vector(blockSums, "blockSums");
	}
}

void opencl_WE_addBlockSums(std::vector<int> &output, std::vector<int> blockSumsScanned)
{
	std::cout << "Adding blocksums ... " << std::endl;
	cl::CommandQueue queue(context, devices[0], 0, &err);
	auto TIME_START = std::chrono::high_resolution_clock::now();
	// INPUT BUFFER
	cl::Buffer output_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, output.size() * sizeof(int));
	// OUTPUT BUFFER
	cl::Buffer blockSumsScanned_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, blockSumsScanned.size() * sizeof(int));

	queue.enqueueWriteBuffer(
		output_buffer, // SOURCE BUFFER
		CL_TRUE, // BLOCK UNTIL OPERATION COMPLETE
		0, // OFFSET
		output.size() * sizeof(int), // SIZE OF ARRAY 
		&output[0] // POINTER TO VECTOR
	);

	queue.enqueueWriteBuffer(
		blockSumsScanned_buffer, // SOURCE BUFFER
		CL_TRUE, // BLOCK UNTIL OPERATION COMPLETE
		0, // OFFSET
		blockSumsScanned.size() * sizeof(int), // SIZE OF ARRAY 
		&blockSumsScanned[0] // POINTER TO VECTOR
	);

	// CREATE KERNEL
	const std::string KERNEL = "addBlockSums";
	cl::Kernel addKernel(program, KERNEL.c_str(), &err);

	// SET KERNEL ARGUMENTS
	addKernel.setArg(0, output_buffer); // PROBLEM ARRAY
	addKernel.setArg(1, blockSumsScanned_buffer); // SCANNED ARRAY

	// SET ND RANGE FOR KERNEL (problem size, local size, global ranges, offset)
	cl::NDRange global(output.size());
	cl::NDRange local(BLOCK_SIZE); //make sure local range is divisible by global range

	std::cout << "CALL KERNEL " << KERNEL << std::endl;
	// ENQUEUE KERNEL EXECUTION
	queue.enqueueNDRangeKernel(addKernel, 0, global, local);

	// PREPARE AND WAIT FOR KERNEL RESULT
	cl::Event event;
	queue.enqueueReadBuffer(
		output_buffer,
		CL_TRUE,
		0,
		output.size() * sizeof(int),
		&output[0],
		NULL,
		&event
	);

	queue.enqueueReadBuffer(
		blockSumsScanned_buffer,
		CL_TRUE,
		0,
		blockSumsScanned.size() * sizeof(int),
		&blockSumsScanned[0],
		NULL,
		&event
	);
	event.wait();
	auto TIME_END = std::chrono::high_resolution_clock::now();
	std::cout << "Received data from kernel" << std::endl;
	auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
	std::cout << "GPU Scan Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
	std::cout << "**********************************************" << std::endl;
	if (PRINT)
		print_vector(output, "Final Scan Result of arbitrary array");

}

std::vector<int> opencl_naive_scan(std::vector<int> vec)
{
	std::vector<int> scanned;
	scanned.resize(vec.size());
	try {
		cl::CommandQueue queue(context, devices[0], 0, &err);

		// INPUT BUFFER
		cl::Buffer source_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, vec.size() * sizeof(int));
		// OUTPUT BUFFER
		cl::Buffer dest_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, vec.size() * sizeof(int));

		auto TIME_START = std::chrono::high_resolution_clock::now();
		// WRITE VECTOR TO GPU BUFFER
		queue.enqueueWriteBuffer(
			source_buffer, // SOURCE BUFFER
			CL_TRUE, // BLOCK UNTIL OPERATION COMPLETE
			0, // OFFSET
			vec.size() * sizeof(int), // SIZE OF ARRAY 
			&vec[0] // POINTER TO VECTOR
		);

		// CREATE KERNEL
		const std::string KERNEL = "scan_parallel_naive_SAT";
		cl::Kernel addKernel(program, KERNEL.c_str(), &err);

		// SET KERNEL ARGUMENTS
		addKernel.setArg(0, source_buffer); // PROBLEM ARRAY
		addKernel.setArg(1, dest_buffer); // SCANNED ARRAY
		addKernel.setArg(2, vec.size()); // PROBLEM SIZE

		// SET ND RANGE FOR KERNEL (problem size, local size, global ranges, offset)
		cl::NDRange global(vec.size());
		cl::NDRange local(vec.size()); //make sure local range is divisible by global range

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
			&scanned[0],
			NULL,
			&event
		);

		// WAITING FOR KERNEL FINISH EVENT
		event.wait();
		auto TIME_END = std::chrono::high_resolution_clock::now();
		std::cout << "Received data from kernel" << std::endl;
		auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
		std::cout << "GPU Scan Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
		std::cout << "**********************************************" << std::endl;
		if (true)
			print_vector(scanned, "Naive Scan Result");
	}
	catch (cl::Error err) {
		print_OpenCl_Error(err);
	}
	return scanned;
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
	if (PRINT)
		print_vector(scanned, "Sequential Scan Result");
	return scanned;
}

void print_vector(std::vector<int> vec, std::string info)
{
	std::cout << info << std::endl;
	int j = 0;
	for (unsigned int i = 0; i < vec.size(); i++)
	{
		std::cout << vec.at(i);
		if (i < vec.size()-1) {
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
	if (PRINT) {
		print_vector(vec, "Original Problem Vector");
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
	// platform = platforms.size() == 2 ? platforms[1] : platforms[0];
	platform = platforms[1];
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
	
	std::cout << "**********************************************" << std::endl;
	std::cout << "Selected Platform Information: " << std::endl;
	std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "Platform Version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
	std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
	std::cout << "**********************************************" << std::endl;
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
