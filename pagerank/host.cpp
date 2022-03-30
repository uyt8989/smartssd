/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

// OpenCL utility layer include
#include "xcl2.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>

using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

const int columns = 2400;
const int rows = 2400;
const int iterations = 100;
const float d = 0.85;

auto constexpr num_cu = 3;

//input : a[row][columns], b[columns] output: b(= a * b);
void matmul(float *a, float *b) {
	//available optimization : store V value first and calculate
	vector<float> temp(columns, 0);

    for(int i = 0; i < rows; i++) {
    	for(int j = 0; j < columns; j++) {
    		temp[i] += a[i * rows + j] * b[j];
    	}
    }

    for(int i = 0; i < columns; i++)
    	b[i] = temp[i];
}

int gen_random() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, 10);

    return dist(e);
}

void print(float* data, int columns, int rows) {
	vector<float> sum(columns, 0);
	for (int r = 0; r < rows; r++) {
		std::cout << r << ": ";
		for (int c = 0; c < columns; c++) {
			sum[c] += data[r * columns + c];
	        std::cout << std::setw(10) << data[r * columns + c] << " ";
	    }
	    std::cout << "\n";
	}

	std::cout << "sum\n";
	for (int c = 0; c < columns; c++) {
		std::cout << std::setw(10) << sum[c] << " ";
	}
	std::cout << "\n\n";
}

void norm(float *data, int columns, int rows) {
	vector<double> sum(columns, 0);
	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < columns; c++) {
			sum[c] += data[r * columns + c];
		}
	}

	for(int r = 0; r < rows; r++){
		for(int c = 0; c < columns; c++) {
			data[r * columns + c] /= sum[c];
		}
	}
}

void verify(vector<float, aligned_allocator<float> >& gold, vector<float, aligned_allocator<float> >& output) {
    for (int i = 0; i < (int)output.size(); i++) {
        if (output[i] != gold[i]) {
            std::cout << "Mismatch " << i << ": gold: " << gold[i] << " device: " << output[i] << "\n";
            print(output.data(), 1, rows);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string binaryFile = argv[1];

    cl_int err;
    cl::CommandQueue q;
    cl::Context context;

    //make kernels
    std::vector<cl::Kernel> krnls(num_cu);

    cl::Program program;

    /*******************************************************************************
    *
   	*	Make data
   	*
  	*******************************************************************************/

    vector<float, aligned_allocator<float>> M(columns * rows);
	vector<float, aligned_allocator<float>> V(columns);
	vector<float, aligned_allocator<float>> C(columns, 0);
	uint32_t repeat_counter = iterations;

    generate(begin(M), end(M), gen_random);
    generate(begin(V), end(V), gen_random);

    norm(M.data(), columns, rows);

    for(int i = 0; i < rows * columns; i++) {
		M[i] = d * M[i] + (1-d) / columns;
	}

    norm(V.data(), 1, rows);
	vector<float, aligned_allocator<float>> gold = { V.begin(), V.end() };

	/*******************************************************************************
	*
	*	Check host execution time
	*
	*******************************************************************************/

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    for(int i = 0; i < iterations; i++)
    	matmul(M.data(), gold.data());
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	std::chrono::nanoseconds nano = end - start;


    /*******************************************************************************
	*
	*	Find Device
	*
    *******************************************************************************/

    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;

    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE |
        		CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        program = cl::Program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_cu; i++) {
    	OCL_CHECK(err, krnls[i] = cl::Kernel(program, "cu3_pagerank", &err));
    }


	std::cout << "|-------------------------+-------------------------|\n"
              << "| Host                    |    Wall-Clock Time (ns) |\n"
              << "|-------------------------+-------------------------|\n";

	std::cout << "|" << std::left << std::setw(24) << "Host: "
              << "|" << std::right << std::setw(24) << nano.count() / repeat_counter << " |\n";

    std::cout << "|-------------------------+-------------------------|\n"
              << "| Kernel                  |    Wall-Clock Time (ns) |\n"
              << "|-------------------------+-------------------------|\n";

    std::vector<cl::Event> event(num_cu);
    uint64_t nstimestart, nstimeend, k_start, k_end;

    /*******************************************************************************
    *
    *	Launch kernel : Pagerank algorithm using multiple compute units
    *
    *******************************************************************************/

    // compute the size of array in bytes
    auto chunk_size = columns * rows / num_cu;
    auto result_size = columns / num_cu;
    size_t mat_size_bytes = chunk_size * sizeof(float);
    size_t vec_size_bytes = columns * sizeof(float);
    size_t out_size_bytes = result_size * sizeof(float);
    uint64_t total_execution_time = 0;

	std::vector<cl::Buffer> buffer_in1(num_cu);
    std::vector<cl::Buffer> buffer_output(num_cu);

	for (int i = 0; i < num_cu; i++) {
    	OCL_CHECK(err, buffer_in1[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, mat_size_bytes,
    		                                                  M.data() + i * chunk_size, &err));
    	OCL_CHECK(err, buffer_output[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, out_size_bytes,
    		                                                C.data() + i * result_size, &err));
    }
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vec_size_bytes, V.data(), &err));

	for (int i = 0; i < num_cu; i++) {
    	// Setting kernel arguments
    	OCL_CHECK(err, err = krnls[i].setArg(0, buffer_in1[i]));
    	OCL_CHECK(err, err = krnls[i].setArg(1, buffer_in2));
    	OCL_CHECK(err, err = krnls[i].setArg(2, buffer_output[i]));
    	OCL_CHECK(err, err = krnls[i].setArg(3, columns));
    	OCL_CHECK(err, err = krnls[i].setArg(4, result_size));
  	}

    for(int iters = 0; iters < iterations; iters++) {
    	for (int i = 0; i < num_cu; i++) {
    	    // Copy input data to device global memory
    	    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1[i], buffer_in2}, 0 /* 0 means from host*/));
    	}
    	OCL_CHECK(err, err = q.finish());

    	for (int i = 0; i < num_cu; i++) {
    		// Launch the kernel
    		OCL_CHECK(err, err = q.enqueueTask(krnls[i], nullptr, &event[i]));
    	}
    	OCL_CHECK(err, err = q.finish());

    	// Copy result from device global memory to host local memory
    	for (int i = 0; i < num_cu; i++) {
    		OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
    	}
    	OCL_CHECK(err, err = q.finish());

    	//Copy the total result to input for next iterations
    	for(int col = 0; col < columns; col++) {
    		V[col] = C[col];
    	}

    	OCL_CHECK(err, err = event[0].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &k_start));
    	OCL_CHECK(err, err = event[0].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &k_end));

    	for(int i = 1; i < num_cu; i++) {
    		OCL_CHECK(err, err = event[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    		OCL_CHECK(err, err = event[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));

    		if(k_start > nstimestart) {
    			k_start = nstimestart;
    		}
    		if(k_end < nstimeend) {
    			k_end = nstimeend;
    		}
    	}
    	total_execution_time += k_end - k_start;

    	std::cout << std::setw(3) << iters << "th time : " << k_end - k_start << "\n";
    }

    verify(gold, C);

    std::cout << "| " << std::left << std::setw(24) << "total : "
              << "|" << std::right << std::setw(24) << total_execution_time << " |\n";
    std::cout << "| " << std::left << std::setw(24) << "avg per iters : "
              << "|" << std::right << std::setw(24) << total_execution_time / repeat_counter << " |\n";
    std::cout << "|-------------------------+-------------------------|\n";
    std::cout << "Note: Wall Clock Time is meaningful for real hardware execution "
              << "only, not for emulation.\n";
    std::cout << "Please refer to profile summary for kernel execution time for "
              << "hardware emulation.\n";
    std::cout << "TEST PASSED\n\n";

    return EXIT_SUCCESS;
}
