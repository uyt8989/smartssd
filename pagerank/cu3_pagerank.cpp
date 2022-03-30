/*******************************************************************************
Description:
   Pagerank algorithm : using multiple compute units (just 1 iteration)
   RES_SIZE = size of pages / number of compute units

*******************************************************************************/

// Includes
#include <stdio.h>
#include <string.h>

#define MAX_SIZE 2400
#define RES_SIZE 800

// TRIPCOUNT identifiers
const unsigned int c_dim = MAX_SIZE;
const unsigned int d_dim = RES_SIZE;

extern "C" {
void cu3_pagerank(float* in1, float* in2, float* out_r, int size, int res_size) {
    // Local buffers to hold temporary data
    float temp_sum[RES_SIZE];
    float B[MAX_SIZE];

// Read data from global memory and write into local buffer for in2
readB:
    for (int itr = 0; itr < size; itr++) {
#pragma HLS LOOP_TRIPCOUNT min = c_dim max = c_dim
#pragma HLS PIPELINE II=1
        B[itr] = in2[itr];
    }

nopart1:
	for (int row = 0; row < res_size; row++) {
#pragma HLS LOOP_TRIPCOUNT min = d_dim max = d_dim
    nopart2:
    	temp_sum[row] = 0;
        for (int col = 0; col < size; col++) {
#pragma HLS LOOP_TRIPCOUNT min = c_dim max = c_dim
#pragma HLS PIPELINE II=1
        nopart3:
        	temp_sum[row] += in1[row * size + col] * B[col];
        }
    }

//	for(int row = 0; row < res_size; row++) {
//#pragma HLS LOOP_TRIPCOUNT min = d_dim max = d_dim
//		in2[row] = temp_sum[row];
//    }

// Write results from local buffer to global memory for out
writeC:
    for (int itr = 0; itr < res_size; itr++) {
#pragma HLS LOOP_TRIPCOUNT min = d_dim max = d_dim
#pragma HLS PIPELINE II=1
        out_r[itr] = temp_sum[itr];
    }
}
}
