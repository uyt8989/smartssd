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
#include "cmdlineparser.h"
#include "xcl2.hpp"
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <ctime>

#define DATA_SIZE 4096
#define INCR_VALUE 10

void p2p_host_to_ssd(int& nvmeFd1, int& nvmeFd2,
                     cl::Context context,
                     cl::CommandQueue q,
                     cl::Program program,
                     std::vector<int, aligned_allocator<int> > source_input_A,
					 std::vector<int, aligned_allocator<int> > source_input_B) {
    int err;
    int ret = 0;
    int size = DATA_SIZE;
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;

    cl::Kernel krnl_vadd;
    // Allocate Buffer in Global Memory
    cl_mem_ext_ptr_t outExt;
    outExt = {XCL_MEM_EXT_P2P_BUFFER, nullptr, 0};

    OCL_CHECK(err, cl::Buffer input_a(context, CL_MEM_READ_ONLY, vector_size_bytes, nullptr, &err));
    OCL_CHECK(err, cl::Buffer input_b(context, CL_MEM_READ_ONLY, vector_size_bytes, nullptr, &err));
    //OCL_CHECK(err, cl::Buffer p2pBo(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, vector_size_bytes, &outExt, &err));
    //OCL_CHECK(err, krnl_vadd = cl::Kernel(program, "adder", &err));

    int* inputPtr_A = (int*)q.enqueueMapBuffer(input_a, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, vector_size_bytes,
                                             nullptr, nullptr, &err);
    int* inputPtr_B = (int*)q.enqueueMapBuffer(input_b, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, vector_size_bytes,
                                                 nullptr, nullptr, &err);

    for (int i = 0; i < DATA_SIZE; i++) {
        inputPtr_A[i] = source_input_A[i];
        inputPtr_B[i] = source_input_B[i];
    }

    q.finish();

    std::cout << "Now start P2P Write from device buffers to SSD\n" << std::endl;
    //save result twice for next step
    ret = pwrite(nvmeFd1, (void*)inputPtr_A, vector_size_bytes, 0);
    if (ret == -1) std::cout << "P2P: write() 1 failed, err: " << ret << ", line: " << __LINE__ << std::endl;
    ret = pwrite(nvmeFd2, (void*)inputPtr_B, vector_size_bytes, 0);
    if (ret == -1) std::cout << "P2P: write() 2 failed, err: " << ret << ", line: " << __LINE__ << std::endl;

    std::cout << "Clean up the buffers\n" << std::endl;
}

void p2p_ssd_to_host(int& nvmeFd1, int &nvmeFd2, int &nvmeFd3,
                     cl::Context context,
                     cl::CommandQueue q,
                     cl::Program program,
                     std::vector<int, aligned_allocator<int> >* source_hw_results) {
    int err, ret = 0;
    int size = DATA_SIZE;
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;

    cl::Kernel krnl_vadd1;
    // Allocate Buffer in Global Memory
    cl_mem_ext_ptr_t inExt1, inExt2, outExt;
    inExt1 = {XCL_MEM_EXT_P2P_BUFFER, nullptr, 0};
    inExt2 = {XCL_MEM_EXT_P2P_BUFFER, nullptr, 0};
    outExt = {0, nullptr, 0};

    //load data(A +B) twice
    OCL_CHECK(err, cl::Buffer buffer_input_a(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, vector_size_bytes, &inExt1,
                                           &err));
    OCL_CHECK(err, cl::Buffer buffer_input_b(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, vector_size_bytes, &inExt2,
                                               &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, vector_size_bytes,
                                            &outExt, &err));
    OCL_CHECK(err, krnl_vadd1 = cl::Kernel(program, "adder", &err));

    std::cout << "\nMap P2P device buffers to host access pointers\n" << std::endl;
    void* p2pPtr1 = q.enqueueMapBuffer(buffer_input_a,      // buffer
                                       CL_TRUE,           // blocking call
                                       CL_MAP_READ,       // Indicates we will be writing
                                       0,                 // buffer offset
                                       vector_size_bytes, // size in bytes
                                       nullptr, nullptr,
                                       &err); // error code
    void* p2pPtr2 = q.enqueueMapBuffer(buffer_input_b,      // buffer
                                          CL_TRUE,           // blocking call
                                          CL_MAP_READ,       // Indicates we will be writing
                                          0,                 // buffer offset
                                          vector_size_bytes, // size in bytes
                                          nullptr, nullptr,
                                          &err); // error code

    std::cout << "Now start P2P Read from SSD to device buffers\n" << std::endl;
    if (pread(nvmeFd1, (void*)p2pPtr1, vector_size_bytes, 0) <= 0) {
        std::cerr << "ERR: pread 1 failed: "
                  << " error: " << strerror(errno) << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<int, aligned_allocator<int> > source_input_A_file(DATA_SIZE);
    std::vector<int, aligned_allocator<int> > source_input_B_file(DATA_SIZE);
    OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_input_a, CL_TRUE, 0, vector_size_bytes, source_input_A_file.data(),
                                                 nullptr, nullptr));
    std::cout << "\ncheck A.txt";
    for(int i = 0; i < DATA_SIZE; i++) {
    	if(i % 256 == 0) std::cout << '\n';
    	std::cout << source_input_A_file[i] << ' ';
    }

    if (pread(nvmeFd2, (void*)p2pPtr2, vector_size_bytes, 0) <= 0) {
          std::cerr << "ERR: pread 2 failed: "
                    << " error: " << strerror(errno) << std::endl;
          exit(EXIT_FAILURE);
    }

    OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_input_b, CL_TRUE, 0, vector_size_bytes, source_input_B_file.data(),
                                                     nullptr, nullptr));

    std::cout << "\ncheck B.txt";
    for(int i = 0; i < DATA_SIZE; i++) {
       	if(i % 256 == 0) std::cout << '\n';
       	std::cout << source_input_B_file[i] << ' ';
    }


    // Set the Kernel Arguments
    OCL_CHECK(err, err = krnl_vadd1.setArg(0, buffer_input_a));
    OCL_CHECK(err, err = krnl_vadd1.setArg(1, buffer_input_b));
    OCL_CHECK(err, err = krnl_vadd1.setArg(2, buffer_output));
    OCL_CHECK(err, err = krnl_vadd1.setArg(3, size));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vadd1));

    // Read output data to host
    // (A + B)
    OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_output, CL_TRUE, 0, vector_size_bytes, source_hw_results->data(),
                                             nullptr, nullptr));
    std::cout << "\nMap P2P device buffers to host access pointers\n" << std::endl;

    std::cout << "\nWrite the result to device\n";
    void* p2pPtr = q.enqueueMapBuffer(buffer_output,                      // buffer
                                      CL_TRUE,                    // blocking call
                                      CL_MAP_WRITE | CL_MAP_READ, // Indicates we will be writing
                                      0,                          // buffer offset
                                      vector_size_bytes,          // size in bytes
                                      nullptr, nullptr,
                                      &err); // error code
    ret = pwrite(nvmeFd3, (void*)p2pPtr, vector_size_bytes, 0);
    if (ret == -1) std::cout << "P2P: write() 3 failed, err: " << ret << ", line: " << __LINE__ << std::endl;

    std::cout << "Clean up the buffers\n" << std::endl;
}

int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--file_path", "-p", "file path string", "");
    parser.addSwitch("--input_file", "-f", "input file string", "");
    parser.addSwitch("--device", "-d", "device id", "0");
    parser.parse(argc, argv);

    // Read settings
    auto binaryFile = parser.value("xclbin_file");
    std::string filepath = parser.value("file_path");
    std::string dev_id = parser.value("device");
    std::string filename;

    if (argc < 5) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    if (filepath.empty()) {
        std::cout << "\nWARNING: As file path is not provided using -p option, going with -f option which is local "
                     "file testing. Please use -p option, if looking for actual p2p operation on NVMe drive.\n";
        filename = parser.value("input_file");
    } else {
        std::cout << "\nWARNING: Ignoring -f option when -p options is set. -p has high precedence over -f.\n";
        filename = filepath;
    }

    int nvmeFd1 = -1, nvmeFd2 = -1, nvmeFd3 = -1;

    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    std::vector<int, aligned_allocator<int> > source_input_A(DATA_SIZE, 3);
    std::vector<int, aligned_allocator<int> > source_input_B(DATA_SIZE, 4);

    /*
    std::srand(unsigned(std::time(0)));
    std::vector<int, aligned_allocator<int>> source_input_A(DATA_SIZE);
    std::generate(source_input_A.begin(), source_input_A.end(), std::rand);
    std::vector<int, aligned_allocator<int>> source_input_B(DATA_SIZE);
    std::generate(source_input_B.begin(), source_input_B.end(), std::rand);

*/
    std::vector<int, aligned_allocator<int> > source_sw_results(DATA_SIZE);
    std::vector<int, aligned_allocator<int> > source_hw_results(DATA_SIZE);

    // Create the test data and Software Result
    for (int i = 0; i < DATA_SIZE; i++) {
    	// (A + B) + (A + B) = 2A + 2B
    	source_sw_results[i] = source_input_A[i] + source_input_B[i];
    }

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    cl::Program program;

    auto pos = dev_id.find(":");
    cl::Device device;
    if (pos == std::string::npos) {
        uint32_t device_index = stoi(dev_id);
        if (device_index >= devices.size()) {
            std::cout << "The device_index provided using -d flag is outside the range of "
                         "available devices\n";
            return EXIT_FAILURE;
        }
        device = devices[device_index];
    } else {
        if (xcl::is_emulation()) {
            std::cout << "Device bdf is not supported for the emulation flow\n";
            return EXIT_FAILURE;
        }
        device = xcl::find_device_bdf(devices, dev_id);
    }

    if (xcl::is_hw_emulation()) {
        auto device_name = device.getInfo<CL_DEVICE_NAME>();
        if (device_name.find("2018") != std::string::npos) {
            std::cout << "[INFO]: The example is not supported for " << device_name
                      << " this platform for hw_emu. Please try other flows." << '\n';
            return EXIT_SUCCESS;
        }
    }

    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << dev_id << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    program = cl::Program(context, {device}, bins, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device[" << dev_id << "] with xclbin file!\n";
        exit(EXIT_FAILURE);
    } else
        std::cout << "Device[" << dev_id << "]: program successful!\n";

    // P2P transfer from host to SSD
    std::cout << "############################################################\n";
    std::cout << "                  Writing data to SSD                       \n";
    std::cout << "############################################################\n";
    // Get access to the NVMe SSD.
    // /dev/nvme0n1 /mnt/csd0/a.txt

    nvmeFd1 = open("/mnt/csd0/A.txt", O_RDWR | O_DIRECT | O_CREAT, 777);
    if (nvmeFd1 < 0) {
    	std::cerr << "ERROR: open " << "/mnt/csd0/A.txt " << "failed: " << std::endl;
    	return EXIT_FAILURE;
    }
    nvmeFd2 = open("/mnt/csd0/B.txt", O_RDWR | O_DIRECT | O_CREAT, 777);
    if (nvmeFd2 < 0) {
    	std::cerr << "ERROR: open " << "/mnt/csd0/B.txt " << "failed: " << std::endl;
    	return EXIT_FAILURE;
    }
    std::cout << "INFO: Successfully opened NVME SSD " << "/mnt/csd0/A.txt, /mnt/csd0/B.txt" << std::endl;
    p2p_host_to_ssd(nvmeFd1, nvmeFd2, context, q, program, source_input_A, source_input_B);
    (void)close(nvmeFd1); (void)close(nvmeFd2);


    // P2P transfer from SSD to host
    std::cout << "############################################################\n";
    std::cout << "                  Reading data from SSD                       \n";
    std::cout << "############################################################\n";

    nvmeFd1 = open("/mnt/csd0/A.txt", O_RDWR | O_DIRECT | O_CREAT, 777);
    if (nvmeFd1 < 0) {
    	std::cerr << "ERROR: open " << "/mnt/csd0/A.txt " << "failed: " << std::endl;
    	return EXIT_FAILURE;
    }
    nvmeFd2 = open("/mnt/csd0/B.txt", O_RDWR | O_DIRECT | O_CREAT, 777);
    if (nvmeFd2 < 0) {
        std::cerr << "ERROR: open " << "/mnt/csd0/B.txt " << "failed: " << std::endl;
        return EXIT_FAILURE;
    }
    nvmeFd3 = open("/mnt/csd0/C.txt", O_RDWR | O_DIRECT | O_CREAT, 777);
    if (nvmeFd3 < 0) {
    	std::cerr << "ERROR: open " << "/mnt/csd0/C.txt " << "failed: " << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "INFO: Successfully opened NVME SSD " << "/mnt/csd0/A.txt, /mnt/csd0/B.txt, /mnt/csd0/C.txt" << std::endl;

    bool num_matched = true;
    p2p_ssd_to_host(nvmeFd1, nvmeFd2, nvmeFd3, context, q, program, &source_hw_results);

    // Validating the results
    if (memcmp(static_cast<void*>(source_sw_results.data()), static_cast<void*>(source_hw_results.data()), DATA_SIZE) !=
        0) {
        num_matched = false;
    }

    (void)close(nvmeFd1); (void)close(nvmeFd2); (void)close(nvmeFd3);


    std::cout << "Check the C.txt";
    std::vector<int, aligned_allocator<int>> source_file_results(DATA_SIZE);
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    cl_mem_ext_ptr_t inExt1 = {XCL_MEM_EXT_P2P_BUFFER, nullptr, 0};

    OCL_CHECK(err, cl::Buffer buffer_check(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, vector_size_bytes, &inExt1,
                                               &err));
    void* check = q.enqueueMapBuffer(buffer_check, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, vector_size_bytes,
                                                nullptr, nullptr, &err);


    nvmeFd3 = open("/mnt/csd0/C.txt", O_RDWR | O_DIRECT);
    if (nvmeFd3 < 0) {
    	std::cerr << "ERROR: open " << "/mnt/csd0/C.txt " << "failed: " << std::endl;
    	return EXIT_FAILURE;
    }

    if (pread(nvmeFd3, check, vector_size_bytes, 0) <= 0) {
    	std::cerr << "ERR: pread 3 failed: " << " error: " << strerror(errno) << std::endl;
        exit(EXIT_FAILURE);
    }
    OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_check, CL_TRUE, 0, vector_size_bytes, source_file_results.data(),
                                                    nullptr, nullptr));

    for(int i = 0; i < DATA_SIZE; i++){
       	if(i % 256 == 0) std::cout << '\n';
       	std::cout << source_file_results[i] << ' ';
    }


    (void)close(nvmeFd3);

    std::cout << "\nTEST " << (num_matched ? "PASSED" : "FAILED") << std::endl;
    return (num_matched ? EXIT_SUCCESS : EXIT_FAILURE);
}
