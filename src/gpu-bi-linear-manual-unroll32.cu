#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// __constant__ float gtop_left_coeff[16];
// __constant__ float gtop_right_coeff[16];
// __constant__ float gbottom_left_coeff[16];
// __constant__ float gbottom_right_coeff[16];

static int INPUTHEIGHT;
static int INPUTWIDTH;

__global__ void base_kernel(float *input, float *output, int x, int y) {
    int thread_id = threadIdx.x; //get thread id; 0-127
    int block_id = blockIdx.x; // get block id;

    // gtop_left_coeff = {1, 2 / 3.0, 1 / 3.0, 0, 2 / 3.0, 4 / 9.0, 2 / 9.0, 0, 1 / 3.0, 2 / 9.0, 1 / 9.0, 0, 0, 0, 0, 0};
    float gtop_left_coeff[16];
    gtop_left_coeff[0] = 1;
    gtop_left_coeff[1] = 2 / 3.0;
    gtop_left_coeff[2] = 1 / 3.0;
    gtop_left_coeff[3] = 0;
    gtop_left_coeff[4] = 2 / 3.0;
    gtop_left_coeff[5] = 4 / 9.0;
    gtop_left_coeff[6] = 2 / 9.0;
    gtop_left_coeff[7] = 0;
    gtop_left_coeff[8] = 1 / 3.0;
    gtop_left_coeff[9] = 2 / 9.0;
    gtop_left_coeff[10] = 1 / 9.0;
    gtop_left_coeff[11] = 0;
    gtop_left_coeff[12] = 0;
    gtop_left_coeff[13] = 0;
    gtop_left_coeff[14] = 0;
    gtop_left_coeff[15] = 0;
    

    // gtop_right_coeff[16] = {0, 1 / 3.0, 2 / 3.0, 1, 0, 2 / 9.0, 4 / 9.0, 2 / 3.0, 0, 1 / 9.0, 2 / 9.0, 1 / 3.0, 0, 0, 0, 0};
    float gtop_right_coeff[16];
    gtop_right_coeff[0] = 0;
    gtop_right_coeff[1] = 1 / 3.0;
    gtop_right_coeff[2] = 2 / 3.0;
    gtop_right_coeff[3] = 1;
    gtop_right_coeff[4] = 0;
    gtop_right_coeff[5] = 2 / 9.0;
    gtop_right_coeff[6] = 4 / 9.0;
    gtop_right_coeff[7] = 2 / 3.0;
    gtop_right_coeff[8] = 0;
    gtop_right_coeff[9] = 1 / 9.0;
    gtop_right_coeff[10] = 2 / 9.0;
    gtop_right_coeff[11] = 1 / 3.0;
    gtop_right_coeff[12] = 0;
    gtop_right_coeff[13] = 0;
    gtop_right_coeff[14] = 0;
    gtop_right_coeff[15] = 0;
    
    // float gbottom_left_coeff[] = {0, 0, 0, 0, 1 / 3.0, 2 / 9.0, 1 / 9.0, 0, 2 / 3.0, 4 / 9.0, 2 / 9.0, 0, 1, 2 / 3.0, 1 / 3.0, 0};
    float gbottom_left_coeff[16];
    gbottom_left_coeff[0] = 0;
    gbottom_left_coeff[1] = 0;
    gbottom_left_coeff[2] = 0;
    gbottom_left_coeff[3] = 0;
    gbottom_left_coeff[4] = 1/3.0;
    gbottom_left_coeff[5] = 2/9.0;
    gbottom_left_coeff[6] = 1/9.0;
    gbottom_left_coeff[7] = 0;
    gbottom_left_coeff[8] = 2 / 3.0;
    gbottom_left_coeff[9] = 4 / 9.0;
    gbottom_left_coeff[10] = 2 / 9.0;
    gbottom_left_coeff[11] = 0;
    gbottom_left_coeff[12] = 1;
    gbottom_left_coeff[13] = 2 / 3.0;
    gbottom_left_coeff[14] = 1 / 3.0;
    gbottom_left_coeff[15] = 0;

    // float gbottom_right_coeff[] = {0, 0, 0, 0, 0, 1 / 9.0, 2 / 9.0, 1 / 3.0, 0, 2 / 9.0, 4 / 9.0, 2 / 3.0, 0, 1 / 3.0, 2 / 3.0, 1};
    float gbottom_right_coeff[16];
    gbottom_right_coeff[0] = 0;
    gbottom_right_coeff[1] = 0;
    gbottom_right_coeff[2] = 0;
    gbottom_right_coeff[3] = 0;
    gbottom_right_coeff[4] = 0;
    gbottom_right_coeff[5] = 1 / 9.0;
    gbottom_right_coeff[6] = 2 / 9.0;
    gbottom_right_coeff[7] = 1 / 3.0;
    gbottom_right_coeff[8] = 2 / 9.0;
    gbottom_right_coeff[9] = 2 / 9.0;
    gbottom_right_coeff[10] = 4 / 9.0;
    gbottom_right_coeff[11] = 2 / 3.0;
    gbottom_right_coeff[12] = 0;
    gbottom_right_coeff[13] = 1 / 3.0;
    gbottom_right_coeff[14] = 2 / 3.0;
    gbottom_right_coeff[15] = 1;
    
    // 16 x 64 -> 32 x 128
    // int kernel_number = thread_id / 4;
    // 4 input pixel: kernel_number * 2, kernel_number * 2 + 1, kernel_number * 2 + x, kernel_number * 2 + x + 1

    int num_blocks = blockDim.x;
    int num_blocks_per_row = x / 128;

    int block_row = block_id / num_blocks_per_row;
    int block_col = block_id % num_blocks_per_row;
    
    int block_start_idx = (block_row * 32) * (x) + block_col * 128; // input: 4 rows of length x per block
    int block_start_idx_out = (block_row * 64) * (x * 2) + block_col * 256; // output: 8 rows of length 2x per block

    // Base Kernel
    int index = thread_id % 4;
    float top_left_1 = input[block_start_idx + (thread_id / 4) * 2];
    float top_right_1 = input[block_start_idx + (thread_id / 4) * 2 + 1];
    float bottom_left_1 = input[block_start_idx + (thread_id / 4) * 2 + x];
    float bottom_right_1 = input[block_start_idx + (thread_id / 4) * 2 + x + 1];

    float top_left_2 = input[block_start_idx + (thread_id / 4) * 2 + 2 * x];
    float top_right_2 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 2 * x];
    float bottom_left_2 = input[block_start_idx + (thread_id / 4) * 2 + x + 2 * x];
    float bottom_right_2 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 2 * x];

    float top_left_3 = input[block_start_idx + (thread_id / 4) * 2 + 4 * x];
    float top_right_3 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 4 * x];
    float bottom_left_3 = input[block_start_idx + (thread_id / 4) * 2 + x + 4 * x];
    float bottom_right_3 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 4 * x];

    float top_left_4 = input[block_start_idx + (thread_id / 4) * 2 + 6 * x];
    float top_right_4 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 6 * x];
    float bottom_left_4 = input[block_start_idx + (thread_id / 4) * 2 + x + 6 * x];
    float bottom_right_4 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 6 * x];

    float top_left_5 = input[block_start_idx + (thread_id / 4) * 2 + 8 * x];
    float top_right_5 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 8 * x];
    float bottom_left_5 = input[block_start_idx + (thread_id / 4) * 2 + x + 8 * x];
    float bottom_right_5 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 8 * x];

    float top_left_6 = input[block_start_idx + (thread_id / 4) * 2 + 10 * x];
    float top_right_6 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 10 * x];
    float bottom_left_6 = input[block_start_idx + (thread_id / 4) * 2 + x + 10 * x];
    float bottom_right_6 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 10 * x];

    float top_left_7 = input[block_start_idx + (thread_id / 4) * 2 + 12 * x];
    float top_right_7 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 12 * x];
    float bottom_left_7 = input[block_start_idx + (thread_id / 4) * 2 + x + 12 * x];
    float bottom_right_7 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 12 * x];

    float top_left_8 = input[block_start_idx + (thread_id / 4) * 2 + 14 * x];
    float top_right_8 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 14 * x];
    float bottom_left_8 = input[block_start_idx + (thread_id / 4) * 2 + x + 14 * x];
    float bottom_right_8 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 14 * x];

    float top_left_9 = input[block_start_idx + (thread_id / 4) * 2 + 16 * x];
    float top_right_9 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 16 * x];
    float bottom_left_9 = input[block_start_idx + (thread_id / 4) * 2 + x + 16 * x];
    float bottom_right_9 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 16 * x];

    float top_left_10 = input[block_start_idx + (thread_id / 4) * 2 + 18 * x];
    float top_right_10 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 18 * x];
    float bottom_left_10 = input[block_start_idx + (thread_id / 4) * 2 + x + 18 * x];
    float bottom_right_10 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 18 * x];

    float top_left_11 = input[block_start_idx + (thread_id / 4) * 2 + 20 * x];
    float top_right_11 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 20 * x];
    float bottom_left_11 = input[block_start_idx + (thread_id / 4) * 2 + x + 20 * x];
    float bottom_right_11 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 20 * x];

    float top_left_12 = input[block_start_idx + (thread_id / 4) * 2 + 22 * x];
    float top_right_12 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 22 * x];
    float bottom_left_12 = input[block_start_idx + (thread_id / 4) * 2 + x + 22 * x];
    float bottom_right_12 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 22 * x];

    float top_left_13 = input[block_start_idx + (thread_id / 4) * 2 + 24 * x];
    float top_right_13 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 24 * x];
    float bottom_left_13 = input[block_start_idx + (thread_id / 4) * 2 + x + 24 * x];
    float bottom_right_13 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 24 * x];

    float top_left_14 = input[block_start_idx + (thread_id / 4) * 2 + 26 * x];
    float top_right_14 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 26 * x];
    float bottom_left_14 = input[block_start_idx + (thread_id / 4) * 2 + x + 26 * x];
    float bottom_right_14 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 26 * x];

    float top_left_15 = input[block_start_idx + (thread_id / 4) * 2 + 28 * x];
    float top_right_15 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 28 * x];
    float bottom_left_15 = input[block_start_idx + (thread_id / 4) * 2 + x + 28 * x];
    float bottom_right_15 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 28 * x];

    float top_left_16 = input[block_start_idx + (thread_id / 4) * 2 + 30 * x];
    float top_right_16 = input[block_start_idx + (thread_id / 4) * 2 + 1 + 30 * x];
    float bottom_left_16 = input[block_start_idx + (thread_id / 4) * 2 + x + 30 * x];
    float bottom_right_16 = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + 30 * x];

    //__device__ float fmaf ( float  x, float  y, float  z )
    float tempOutRow1 = 0.0f;
    float tempOutRow2 = 0.0f;
    float tempOutRow3 = 0.0f;
    float tempOutRow4 = 0.0f;
    float tempOutRow5 = 0.0f;
    float tempOutRow6 = 0.0f;
    float tempOutRow7 = 0.0f;
    float tempOutRow8 = 0.0f;
    float tempOutRow9 = 0.0f;
    float tempOutRow10 = 0.0f;
    float tempOutRow11 = 0.0f;
    float tempOutRow12 = 0.0f;
    float tempOutRow13 = 0.0f;
    float tempOutRow14 = 0.0f;
    float tempOutRow15 = 0.0f;
    float tempOutRow16 = 0.0f;

    float tempOutRow17 = 0.0f;
    float tempOutRow18 = 0.0f;
    float tempOutRow19 = 0.0f;
    float tempOutRow20 = 0.0f;
    float tempOutRow21 = 0.0f;
    float tempOutRow22 = 0.0f;
    float tempOutRow23 = 0.0f;
    float tempOutRow24 = 0.0f;
    float tempOutRow25 = 0.0f;
    float tempOutRow26 = 0.0f;
    float tempOutRow27 = 0.0f;
    float tempOutRow28 = 0.0f;
    float tempOutRow29 = 0.0f;
    float tempOutRow30 = 0.0f;
    float tempOutRow31 = 0.0f;
    float tempOutRow32 = 0.0f;

    float tempOutRow33 = 0.0f;
    float tempOutRow34 = 0.0f;
    float tempOutRow35 = 0.0f;
    float tempOutRow36 = 0.0f;
    float tempOutRow37 = 0.0f;
    float tempOutRow38 = 0.0f;
    float tempOutRow39 = 0.0f;
    float tempOutRow40 = 0.0f;
    float tempOutRow41 = 0.0f;
    float tempOutRow42 = 0.0f;
    float tempOutRow43 = 0.0f;
    float tempOutRow44 = 0.0f;
    float tempOutRow45 = 0.0f;
    float tempOutRow46 = 0.0f;
    float tempOutRow47 = 0.0f;
    float tempOutRow48 = 0.0f;
    
    float tempOutRow49 = 0.0f;
    float tempOutRow50 = 0.0f;
    float tempOutRow51 = 0.0f;
    float tempOutRow52 = 0.0f;
    float tempOutRow53 = 0.0f;
    float tempOutRow54 = 0.0f;
    float tempOutRow55 = 0.0f;
    float tempOutRow56 = 0.0f;
    float tempOutRow57 = 0.0f;
    float tempOutRow58 = 0.0f;
    float tempOutRow59 = 0.0f;
    float tempOutRow60 = 0.0f;
    float tempOutRow61 = 0.0f;
    float tempOutRow62 = 0.0f;
    float tempOutRow63 = 0.0f;
    float tempOutRow64 = 0.0f;

    // Top_Left Partial Sums for all rows
    tempOutRow1 = fmaf(top_left_1, gtop_left_coeff[index], tempOutRow1);
    tempOutRow2 = fmaf(top_left_1, gtop_left_coeff[index + 4], tempOutRow2);
    tempOutRow3 = fmaf(top_left_1, gtop_left_coeff[index + 8], tempOutRow3);
    tempOutRow4 = fmaf(top_left_1, gtop_left_coeff[index + 12], tempOutRow4);
    tempOutRow5 = fmaf(top_left_2, gtop_left_coeff[index], tempOutRow5);
    tempOutRow6 = fmaf(top_left_2, gtop_left_coeff[index + 4], tempOutRow6);
    tempOutRow7 = fmaf(top_left_2, gtop_left_coeff[index + 8], tempOutRow7);
    tempOutRow8 = fmaf(top_left_2, gtop_left_coeff[index + 12], tempOutRow8);
    tempOutRow9 = fmaf(top_left_3, gtop_left_coeff[index], tempOutRow9);
    tempOutRow10 = fmaf(top_left_3, gtop_left_coeff[index + 4], tempOutRow10);
    tempOutRow11 = fmaf(top_left_3, gtop_left_coeff[index + 8], tempOutRow11);
    tempOutRow12 = fmaf(top_left_3, gtop_left_coeff[index + 12], tempOutRow12);
    tempOutRow13 = fmaf(top_left_4, gtop_left_coeff[index], tempOutRow13);
    tempOutRow14 = fmaf(top_left_4, gtop_left_coeff[index + 4], tempOutRow14);
    tempOutRow15 = fmaf(top_left_4, gtop_left_coeff[index + 8], tempOutRow15);
    tempOutRow16 = fmaf(top_left_4, gtop_left_coeff[index + 12], tempOutRow16);
    
    tempOutRow17 = fmaf(top_left_5, gtop_left_coeff[index], tempOutRow17);
    tempOutRow18 = fmaf(top_left_5, gtop_left_coeff[index + 4], tempOutRow18);
    tempOutRow19 = fmaf(top_left_5, gtop_left_coeff[index + 8], tempOutRow19);
    tempOutRow20 = fmaf(top_left_5, gtop_left_coeff[index + 12], tempOutRow20);
    tempOutRow21 = fmaf(top_left_6, gtop_left_coeff[index], tempOutRow21);
    tempOutRow22 = fmaf(top_left_6, gtop_left_coeff[index + 4], tempOutRow22);
    tempOutRow23 = fmaf(top_left_6, gtop_left_coeff[index + 8], tempOutRow23);
    tempOutRow24 = fmaf(top_left_6, gtop_left_coeff[index + 12], tempOutRow24);
    tempOutRow25 = fmaf(top_left_7, gtop_left_coeff[index], tempOutRow25);
    tempOutRow26 = fmaf(top_left_7, gtop_left_coeff[index + 4], tempOutRow26);
    tempOutRow27 = fmaf(top_left_7, gtop_left_coeff[index + 8], tempOutRow27);
    tempOutRow28 = fmaf(top_left_7, gtop_left_coeff[index + 12], tempOutRow28);
    tempOutRow29 = fmaf(top_left_8, gtop_left_coeff[index], tempOutRow29);
    tempOutRow30 = fmaf(top_left_8, gtop_left_coeff[index + 4], tempOutRow30);
    tempOutRow31 = fmaf(top_left_8, gtop_left_coeff[index + 8], tempOutRow31);
    tempOutRow32 = fmaf(top_left_8, gtop_left_coeff[index + 12], tempOutRow32);

    tempOutRow33 = fmaf(top_left_9, gtop_left_coeff[index], tempOutRow33);
    tempOutRow34 = fmaf(top_left_9, gtop_left_coeff[index + 4], tempOutRow34);
    tempOutRow35 = fmaf(top_left_9, gtop_left_coeff[index + 8], tempOutRow35);
    tempOutRow36 = fmaf(top_left_9, gtop_left_coeff[index + 12], tempOutRow36);
    tempOutRow37 = fmaf(top_left_10, gtop_left_coeff[index], tempOutRow37);
    tempOutRow38 = fmaf(top_left_10, gtop_left_coeff[index + 4], tempOutRow38);
    tempOutRow39 = fmaf(top_left_10, gtop_left_coeff[index + 8], tempOutRow39);
    tempOutRow40 = fmaf(top_left_10, gtop_left_coeff[index + 12], tempOutRow40);
    tempOutRow41 = fmaf(top_left_11, gtop_left_coeff[index], tempOutRow41);
    tempOutRow42 = fmaf(top_left_11, gtop_left_coeff[index + 4], tempOutRow42);
    tempOutRow43 = fmaf(top_left_11, gtop_left_coeff[index + 8], tempOutRow43);
    tempOutRow44 = fmaf(top_left_11, gtop_left_coeff[index + 12], tempOutRow44);
    tempOutRow45 = fmaf(top_left_12, gtop_left_coeff[index], tempOutRow45);
    tempOutRow46 = fmaf(top_left_12, gtop_left_coeff[index + 4], tempOutRow46);
    tempOutRow47 = fmaf(top_left_12, gtop_left_coeff[index + 8], tempOutRow47);
    tempOutRow48 = fmaf(top_left_12, gtop_left_coeff[index + 12], tempOutRow48);
    
    tempOutRow49 = fmaf(top_left_13, gtop_left_coeff[index], tempOutRow49);
    tempOutRow50 = fmaf(top_left_13, gtop_left_coeff[index + 4], tempOutRow50);
    tempOutRow51 = fmaf(top_left_13, gtop_left_coeff[index + 8], tempOutRow51);
    tempOutRow52 = fmaf(top_left_13, gtop_left_coeff[index + 12], tempOutRow52);
    tempOutRow53 = fmaf(top_left_14, gtop_left_coeff[index], tempOutRow53);
    tempOutRow54 = fmaf(top_left_14, gtop_left_coeff[index + 4], tempOutRow54);
    tempOutRow55 = fmaf(top_left_14, gtop_left_coeff[index + 8], tempOutRow55);
    tempOutRow56 = fmaf(top_left_14, gtop_left_coeff[index + 12], tempOutRow56);
    tempOutRow57 = fmaf(top_left_15, gtop_left_coeff[index], tempOutRow57);
    tempOutRow58 = fmaf(top_left_15, gtop_left_coeff[index + 4], tempOutRow58);
    tempOutRow59 = fmaf(top_left_15, gtop_left_coeff[index + 8], tempOutRow59);
    tempOutRow60 = fmaf(top_left_15, gtop_left_coeff[index + 12], tempOutRow60);
    tempOutRow61 = fmaf(top_left_16, gtop_left_coeff[index], tempOutRow61);
    tempOutRow62 = fmaf(top_left_16, gtop_left_coeff[index + 4], tempOutRow62);
    tempOutRow63 = fmaf(top_left_16, gtop_left_coeff[index + 8], tempOutRow63);
    tempOutRow64 = fmaf(top_left_16, gtop_left_coeff[index + 12], tempOutRow64);

    // Top_Right Partial Sums for all rows
    tempOutRow1 = fmaf(top_right_1, gtop_right_coeff[index], tempOutRow1);
    tempOutRow2 = fmaf(top_right_1, gtop_right_coeff[index + 4], tempOutRow2);
    tempOutRow3 = fmaf(top_right_1, gtop_right_coeff[index + 8], tempOutRow3);
    tempOutRow4 = fmaf(top_right_1, gtop_right_coeff[index + 12], tempOutRow4);
    tempOutRow5 = fmaf(top_right_2, gtop_right_coeff[index], tempOutRow5);
    tempOutRow6 = fmaf(top_right_2, gtop_right_coeff[index + 4], tempOutRow6);
    tempOutRow7 = fmaf(top_right_2, gtop_right_coeff[index + 8], tempOutRow7);
    tempOutRow8 = fmaf(top_right_2, gtop_right_coeff[index + 12], tempOutRow8);
    tempOutRow9 = fmaf(top_right_3, gtop_right_coeff[index], tempOutRow9);
    tempOutRow10 = fmaf(top_right_3, gtop_right_coeff[index + 4], tempOutRow10);
    tempOutRow11 = fmaf(top_right_3, gtop_right_coeff[index + 8], tempOutRow11);
    tempOutRow12 = fmaf(top_right_3, gtop_right_coeff[index + 12], tempOutRow12);
    tempOutRow13 = fmaf(top_right_4, gtop_right_coeff[index], tempOutRow13);
    tempOutRow14 = fmaf(top_right_4, gtop_right_coeff[index + 4], tempOutRow14);
    tempOutRow15 = fmaf(top_right_4, gtop_right_coeff[index + 8], tempOutRow15);
    tempOutRow16 = fmaf(top_right_4, gtop_right_coeff[index + 12], tempOutRow16);
    
    tempOutRow17 = fmaf(top_right_5, gtop_right_coeff[index], tempOutRow17);
    tempOutRow18 = fmaf(top_right_5, gtop_right_coeff[index + 4], tempOutRow18);
    tempOutRow19 = fmaf(top_right_5, gtop_right_coeff[index + 8], tempOutRow19);
    tempOutRow20 = fmaf(top_right_5, gtop_right_coeff[index + 12], tempOutRow20);
    tempOutRow21 = fmaf(top_right_6, gtop_right_coeff[index], tempOutRow21);
    tempOutRow22 = fmaf(top_right_6, gtop_right_coeff[index + 4], tempOutRow22);
    tempOutRow23 = fmaf(top_right_6, gtop_right_coeff[index + 8], tempOutRow23);
    tempOutRow24 = fmaf(top_right_6, gtop_right_coeff[index + 12], tempOutRow24);
    tempOutRow25 = fmaf(top_right_7, gtop_right_coeff[index], tempOutRow25);
    tempOutRow26 = fmaf(top_right_7, gtop_right_coeff[index + 4], tempOutRow26);
    tempOutRow27 = fmaf(top_right_7, gtop_right_coeff[index + 8], tempOutRow27);
    tempOutRow28 = fmaf(top_right_7, gtop_right_coeff[index + 12], tempOutRow28);
    tempOutRow29 = fmaf(top_right_8, gtop_right_coeff[index], tempOutRow29);
    tempOutRow30 = fmaf(top_right_8, gtop_right_coeff[index + 4], tempOutRow30);
    tempOutRow31 = fmaf(top_right_8, gtop_right_coeff[index + 8], tempOutRow31);
    tempOutRow32 = fmaf(top_right_8, gtop_right_coeff[index + 12], tempOutRow32);

    tempOutRow33 = fmaf(top_right_9, gtop_right_coeff[index], tempOutRow33);
    tempOutRow34 = fmaf(top_right_9, gtop_right_coeff[index + 4], tempOutRow34);
    tempOutRow35 = fmaf(top_right_9, gtop_right_coeff[index + 8], tempOutRow35);
    tempOutRow36 = fmaf(top_right_9, gtop_right_coeff[index + 12], tempOutRow36);
    tempOutRow37 = fmaf(top_right_10, gtop_right_coeff[index], tempOutRow37);
    tempOutRow38 = fmaf(top_right_10, gtop_right_coeff[index + 4], tempOutRow38);
    tempOutRow39 = fmaf(top_right_10, gtop_right_coeff[index + 8], tempOutRow39);
    tempOutRow40 = fmaf(top_right_10, gtop_right_coeff[index + 12], tempOutRow40);
    tempOutRow41 = fmaf(top_right_11, gtop_right_coeff[index], tempOutRow41);
    tempOutRow42 = fmaf(top_right_11, gtop_right_coeff[index + 4], tempOutRow42);
    tempOutRow43 = fmaf(top_right_11, gtop_right_coeff[index + 8], tempOutRow43);
    tempOutRow44 = fmaf(top_right_11, gtop_right_coeff[index + 12], tempOutRow44);
    tempOutRow45 = fmaf(top_right_12, gtop_right_coeff[index], tempOutRow45);
    tempOutRow46 = fmaf(top_right_12, gtop_right_coeff[index + 4], tempOutRow46);
    tempOutRow47 = fmaf(top_right_12, gtop_right_coeff[index + 8], tempOutRow47);
    tempOutRow48 = fmaf(top_right_12, gtop_right_coeff[index + 12], tempOutRow48);
    
    tempOutRow49 = fmaf(top_right_13, gtop_right_coeff[index], tempOutRow49);
    tempOutRow50 = fmaf(top_right_13, gtop_right_coeff[index + 4], tempOutRow50);
    tempOutRow51 = fmaf(top_right_13, gtop_right_coeff[index + 8], tempOutRow51);
    tempOutRow52 = fmaf(top_right_13, gtop_right_coeff[index + 12], tempOutRow52);
    tempOutRow53 = fmaf(top_right_14, gtop_right_coeff[index], tempOutRow53);
    tempOutRow54 = fmaf(top_right_14, gtop_right_coeff[index + 4], tempOutRow54);
    tempOutRow55 = fmaf(top_right_14, gtop_right_coeff[index + 8], tempOutRow55);
    tempOutRow56 = fmaf(top_right_14, gtop_right_coeff[index + 12], tempOutRow56);
    tempOutRow57 = fmaf(top_right_15, gtop_right_coeff[index], tempOutRow57);
    tempOutRow58 = fmaf(top_right_15, gtop_right_coeff[index + 4], tempOutRow58);
    tempOutRow59 = fmaf(top_right_15, gtop_right_coeff[index + 8], tempOutRow59);
    tempOutRow60 = fmaf(top_right_15, gtop_right_coeff[index + 12], tempOutRow60);
    tempOutRow61 = fmaf(top_right_16, gtop_right_coeff[index], tempOutRow61);
    tempOutRow62 = fmaf(top_right_16, gtop_right_coeff[index + 4], tempOutRow62);
    tempOutRow63 = fmaf(top_right_16, gtop_right_coeff[index + 8], tempOutRow63);
    tempOutRow64 = fmaf(top_right_16, gtop_right_coeff[index + 12], tempOutRow64);

    // Bottom_Left Partial Sums for all rows
    tempOutRow1 = fmaf(bottom_left_1, gbottom_left_coeff[index], tempOutRow1);
    tempOutRow2 = fmaf(bottom_left_1, gbottom_left_coeff[index + 4], tempOutRow2);
    tempOutRow3 = fmaf(bottom_left_1, gbottom_left_coeff[index + 8], tempOutRow3);
    tempOutRow4 = fmaf(bottom_left_1, gbottom_left_coeff[index + 12], tempOutRow4);
    tempOutRow5 = fmaf(bottom_left_2, gbottom_left_coeff[index], tempOutRow5);
    tempOutRow6 = fmaf(bottom_left_2, gbottom_left_coeff[index + 4], tempOutRow6);
    tempOutRow7 = fmaf(bottom_left_2, gbottom_left_coeff[index + 8], tempOutRow7);
    tempOutRow8 = fmaf(bottom_left_2, gbottom_left_coeff[index + 12], tempOutRow8);
    tempOutRow9 = fmaf(bottom_left_3, gbottom_left_coeff[index], tempOutRow9);
    tempOutRow10 = fmaf(bottom_left_3, gbottom_left_coeff[index + 4], tempOutRow10);
    tempOutRow11 = fmaf(bottom_left_3, gbottom_left_coeff[index + 8], tempOutRow11);
    tempOutRow12 = fmaf(bottom_left_3, gbottom_left_coeff[index + 12], tempOutRow12);
    tempOutRow13 = fmaf(bottom_left_4, gbottom_left_coeff[index], tempOutRow13);
    tempOutRow14 = fmaf(bottom_left_4, gbottom_left_coeff[index + 4], tempOutRow14);
    tempOutRow15 = fmaf(bottom_left_4, gbottom_left_coeff[index + 8], tempOutRow15);
    tempOutRow16 = fmaf(bottom_left_4, gbottom_left_coeff[index + 12], tempOutRow16);

    tempOutRow17 = fmaf(bottom_left_5, gbottom_left_coeff[index], tempOutRow17);
    tempOutRow18 = fmaf(bottom_left_5, gbottom_left_coeff[index + 4], tempOutRow18);
    tempOutRow19 = fmaf(bottom_left_5, gbottom_left_coeff[index + 8], tempOutRow19);
    tempOutRow20 = fmaf(bottom_left_5, gbottom_left_coeff[index + 12], tempOutRow20);
    tempOutRow21 = fmaf(bottom_left_6, gbottom_left_coeff[index], tempOutRow21);
    tempOutRow22 = fmaf(bottom_left_6, gbottom_left_coeff[index + 4], tempOutRow22);
    tempOutRow23 = fmaf(bottom_left_6, gbottom_left_coeff[index + 8], tempOutRow23);
    tempOutRow24 = fmaf(bottom_left_6, gbottom_left_coeff[index + 12], tempOutRow24);
    tempOutRow25 = fmaf(bottom_left_7, gbottom_left_coeff[index], tempOutRow25);
    tempOutRow26 = fmaf(bottom_left_7, gbottom_left_coeff[index + 4], tempOutRow26);
    tempOutRow27 = fmaf(bottom_left_7, gbottom_left_coeff[index + 8], tempOutRow27);
    tempOutRow28 = fmaf(bottom_left_7, gbottom_left_coeff[index + 12], tempOutRow28);
    tempOutRow29 = fmaf(bottom_left_8, gbottom_left_coeff[index], tempOutRow29);
    tempOutRow30 = fmaf(bottom_left_8, gbottom_left_coeff[index + 4], tempOutRow30);
    tempOutRow31 = fmaf(bottom_left_8, gbottom_left_coeff[index + 8], tempOutRow31);
    tempOutRow32 = fmaf(bottom_left_8, gbottom_left_coeff[index + 12], tempOutRow32);

    tempOutRow33 = fmaf(bottom_left_9, gbottom_left_coeff[index], tempOutRow33);
    tempOutRow34 = fmaf(bottom_left_9, gbottom_left_coeff[index + 4], tempOutRow34);
    tempOutRow35 = fmaf(bottom_left_9, gbottom_left_coeff[index + 8], tempOutRow35);
    tempOutRow36 = fmaf(bottom_left_9, gbottom_left_coeff[index + 12], tempOutRow36);
    tempOutRow37 = fmaf(bottom_left_10, gbottom_left_coeff[index], tempOutRow37);
    tempOutRow38 = fmaf(bottom_left_10, gbottom_left_coeff[index + 4], tempOutRow38);
    tempOutRow39 = fmaf(bottom_left_10, gbottom_left_coeff[index + 8], tempOutRow39);
    tempOutRow40 = fmaf(bottom_left_10, gbottom_left_coeff[index + 12], tempOutRow40);
    tempOutRow41 = fmaf(bottom_left_11, gbottom_left_coeff[index], tempOutRow41);
    tempOutRow42 = fmaf(bottom_left_11, gbottom_left_coeff[index + 4], tempOutRow42);
    tempOutRow43 = fmaf(bottom_left_11, gbottom_left_coeff[index + 8], tempOutRow43);
    tempOutRow44 = fmaf(bottom_left_11, gbottom_left_coeff[index + 12], tempOutRow44);
    tempOutRow45 = fmaf(bottom_left_12, gbottom_left_coeff[index], tempOutRow45);
    tempOutRow46 = fmaf(bottom_left_12, gbottom_left_coeff[index + 4], tempOutRow46);
    tempOutRow47 = fmaf(bottom_left_12, gbottom_left_coeff[index + 8], tempOutRow47);
    tempOutRow48 = fmaf(bottom_left_12, gbottom_left_coeff[index + 12], tempOutRow48);
    
    tempOutRow49 = fmaf(bottom_left_13, gbottom_left_coeff[index], tempOutRow49);
    tempOutRow50 = fmaf(bottom_left_13, gbottom_left_coeff[index + 4], tempOutRow50);
    tempOutRow51 = fmaf(bottom_left_13, gbottom_left_coeff[index + 8], tempOutRow51);
    tempOutRow52 = fmaf(bottom_left_13, gbottom_left_coeff[index + 12], tempOutRow52);
    tempOutRow53 = fmaf(bottom_left_14, gbottom_left_coeff[index], tempOutRow53);
    tempOutRow54 = fmaf(bottom_left_14, gbottom_left_coeff[index + 4], tempOutRow54);
    tempOutRow55 = fmaf(bottom_left_14, gbottom_left_coeff[index + 8], tempOutRow55);
    tempOutRow56 = fmaf(bottom_left_14, gbottom_left_coeff[index + 12], tempOutRow56);
    tempOutRow57 = fmaf(bottom_left_15, gbottom_left_coeff[index], tempOutRow57);
    tempOutRow58 = fmaf(bottom_left_15, gbottom_left_coeff[index + 4], tempOutRow58);
    tempOutRow59 = fmaf(bottom_left_15, gbottom_left_coeff[index + 8], tempOutRow59);
    tempOutRow60 = fmaf(bottom_left_15, gbottom_left_coeff[index + 12], tempOutRow60);
    tempOutRow61 = fmaf(bottom_left_16, gbottom_left_coeff[index], tempOutRow61);
    tempOutRow62 = fmaf(bottom_left_16, gbottom_left_coeff[index + 4], tempOutRow62);
    tempOutRow63 = fmaf(bottom_left_16, gbottom_left_coeff[index + 8], tempOutRow63);
    tempOutRow64 = fmaf(bottom_left_16, gbottom_left_coeff[index + 12], tempOutRow64);

    // Bottom_Right Partial Sums for all rows
    tempOutRow1 = fmaf(bottom_right_1, gbottom_right_coeff[index], tempOutRow1);
    tempOutRow2 = fmaf(bottom_right_1, gbottom_right_coeff[index + 4], tempOutRow2);
    tempOutRow3 = fmaf(bottom_right_1, gbottom_right_coeff[index + 8], tempOutRow3);
    tempOutRow4 = fmaf(bottom_right_1, gbottom_right_coeff[index + 12], tempOutRow4);
    tempOutRow5 = fmaf(bottom_right_2, gbottom_right_coeff[index], tempOutRow5);
    tempOutRow6 = fmaf(bottom_right_2, gbottom_right_coeff[index + 4], tempOutRow6);
    tempOutRow7 = fmaf(bottom_right_2, gbottom_right_coeff[index + 8], tempOutRow7);
    tempOutRow8 = fmaf(bottom_right_2, gbottom_right_coeff[index + 12], tempOutRow8);
    tempOutRow9 = fmaf(bottom_right_3, gbottom_right_coeff[index], tempOutRow9);
    tempOutRow10 = fmaf(bottom_right_3, gbottom_right_coeff[index + 4], tempOutRow10);
    tempOutRow11 = fmaf(bottom_right_3, gbottom_right_coeff[index + 8], tempOutRow11);
    tempOutRow12 = fmaf(bottom_right_3, gbottom_right_coeff[index + 12], tempOutRow12);
    tempOutRow13 = fmaf(bottom_right_4, gbottom_right_coeff[index], tempOutRow13);
    tempOutRow14 = fmaf(bottom_right_4, gbottom_right_coeff[index + 4], tempOutRow14);
    tempOutRow15 = fmaf(bottom_right_4, gbottom_right_coeff[index + 8], tempOutRow15);
    tempOutRow16 = fmaf(bottom_right_4, gbottom_right_coeff[index + 12], tempOutRow16);

    tempOutRow17 = fmaf(bottom_right_5, gbottom_right_coeff[index], tempOutRow17);
    tempOutRow18 = fmaf(bottom_right_5, gbottom_right_coeff[index + 4], tempOutRow18);
    tempOutRow19 = fmaf(bottom_right_5, gbottom_right_coeff[index + 8], tempOutRow19);
    tempOutRow20 = fmaf(bottom_right_5, gbottom_right_coeff[index + 12], tempOutRow20);
    tempOutRow21 = fmaf(bottom_right_6, gbottom_right_coeff[index], tempOutRow21);
    tempOutRow22 = fmaf(bottom_right_6, gbottom_right_coeff[index + 4], tempOutRow22);
    tempOutRow23 = fmaf(bottom_right_6, gbottom_right_coeff[index + 8], tempOutRow23);
    tempOutRow24 = fmaf(bottom_right_6, gbottom_right_coeff[index + 12], tempOutRow24);
    tempOutRow25 = fmaf(bottom_right_7, gbottom_right_coeff[index], tempOutRow25);
    tempOutRow26 = fmaf(bottom_right_7, gbottom_right_coeff[index + 4], tempOutRow26);
    tempOutRow27 = fmaf(bottom_right_7, gbottom_right_coeff[index + 8], tempOutRow27);
    tempOutRow28 = fmaf(bottom_right_7, gbottom_right_coeff[index + 12], tempOutRow28);
    tempOutRow29 = fmaf(bottom_right_8, gbottom_right_coeff[index], tempOutRow29);
    tempOutRow30 = fmaf(bottom_right_8, gbottom_right_coeff[index + 4], tempOutRow30);
    tempOutRow31 = fmaf(bottom_right_8, gbottom_right_coeff[index + 8], tempOutRow31);
    tempOutRow32 = fmaf(bottom_right_8, gbottom_right_coeff[index + 12], tempOutRow32);

    tempOutRow33 = fmaf(bottom_right_9, gbottom_right_coeff[index], tempOutRow33);
    tempOutRow34 = fmaf(bottom_right_9, gbottom_right_coeff[index + 4], tempOutRow34);
    tempOutRow35 = fmaf(bottom_right_9, gbottom_right_coeff[index + 8], tempOutRow35);
    tempOutRow36 = fmaf(bottom_right_9, gbottom_right_coeff[index + 12], tempOutRow36);
    tempOutRow37 = fmaf(bottom_right_10, gbottom_right_coeff[index], tempOutRow37);
    tempOutRow38 = fmaf(bottom_right_10, gbottom_right_coeff[index + 4], tempOutRow38);
    tempOutRow39 = fmaf(bottom_right_10, gbottom_right_coeff[index + 8], tempOutRow39);
    tempOutRow40 = fmaf(bottom_right_10, gbottom_right_coeff[index + 12], tempOutRow40);
    tempOutRow41 = fmaf(bottom_right_11, gbottom_right_coeff[index], tempOutRow41);
    tempOutRow42 = fmaf(bottom_right_11, gbottom_right_coeff[index + 4], tempOutRow42);
    tempOutRow43 = fmaf(bottom_right_11, gbottom_right_coeff[index + 8], tempOutRow43);
    tempOutRow44 = fmaf(bottom_right_11, gbottom_right_coeff[index + 12], tempOutRow44);
    tempOutRow45 = fmaf(bottom_right_12, gbottom_right_coeff[index], tempOutRow45);
    tempOutRow46 = fmaf(bottom_right_12, gbottom_right_coeff[index + 4], tempOutRow46);
    tempOutRow47 = fmaf(bottom_right_12, gbottom_right_coeff[index + 8], tempOutRow47);
    tempOutRow48 = fmaf(bottom_right_12, gbottom_right_coeff[index + 12], tempOutRow48);
    
    tempOutRow49 = fmaf(bottom_right_13, gbottom_right_coeff[index], tempOutRow49);
    tempOutRow50 = fmaf(bottom_right_13, gbottom_right_coeff[index + 4], tempOutRow50);
    tempOutRow51 = fmaf(bottom_right_13, gbottom_right_coeff[index + 8], tempOutRow51);
    tempOutRow52 = fmaf(bottom_right_13, gbottom_right_coeff[index + 12], tempOutRow52);
    tempOutRow53 = fmaf(bottom_right_14, gbottom_right_coeff[index], tempOutRow53);
    tempOutRow54 = fmaf(bottom_right_14, gbottom_right_coeff[index + 4], tempOutRow54);
    tempOutRow55 = fmaf(bottom_right_14, gbottom_right_coeff[index + 8], tempOutRow55);
    tempOutRow56 = fmaf(bottom_right_14, gbottom_right_coeff[index + 12], tempOutRow56);
    tempOutRow57 = fmaf(bottom_right_15, gbottom_right_coeff[index], tempOutRow57);
    tempOutRow58 = fmaf(bottom_right_15, gbottom_right_coeff[index + 4], tempOutRow58);
    tempOutRow59 = fmaf(bottom_right_15, gbottom_right_coeff[index + 8], tempOutRow59);
    tempOutRow60 = fmaf(bottom_right_15, gbottom_right_coeff[index + 12], tempOutRow60);
    tempOutRow61 = fmaf(bottom_right_16, gbottom_right_coeff[index], tempOutRow61);
    tempOutRow62 = fmaf(bottom_right_16, gbottom_right_coeff[index + 4], tempOutRow62);
    tempOutRow63 = fmaf(bottom_right_16, gbottom_right_coeff[index + 8], tempOutRow63);
    tempOutRow64 = fmaf(bottom_right_16, gbottom_right_coeff[index + 12], tempOutRow64);

    output[block_start_idx_out + thread_id] = tempOutRow1;
    output[block_start_idx_out + thread_id + 2*x] = tempOutRow2;
    output[block_start_idx_out + thread_id + 4*x] = tempOutRow3;
    output[block_start_idx_out + thread_id + 6*x] = tempOutRow4;
    output[block_start_idx_out + thread_id + 8*x] = tempOutRow5;
    output[block_start_idx_out + thread_id + 10*x] = tempOutRow6;
    output[block_start_idx_out + thread_id + 12*x] = tempOutRow7;
    output[block_start_idx_out + thread_id + 14*x] = tempOutRow8;
    output[block_start_idx_out + thread_id + 16*x] = tempOutRow9;
    output[block_start_idx_out + thread_id + 18*x] = tempOutRow10;
    output[block_start_idx_out + thread_id + 20*x] = tempOutRow11;
    output[block_start_idx_out + thread_id + 22*x] = tempOutRow12;
    output[block_start_idx_out + thread_id + 24*x] = tempOutRow13;
    output[block_start_idx_out + thread_id + 26*x] = tempOutRow14;
    output[block_start_idx_out + thread_id + 28*x] = tempOutRow15;
    output[block_start_idx_out + thread_id + 30*x] = tempOutRow16;
    
    output[block_start_idx_out + thread_id + 32*x] = tempOutRow17;
    output[block_start_idx_out + thread_id + 34*x] = tempOutRow18;
    output[block_start_idx_out + thread_id + 36*x] = tempOutRow19;
    output[block_start_idx_out + thread_id + 38*x] = tempOutRow20;
    output[block_start_idx_out + thread_id + 40*x] = tempOutRow21;
    output[block_start_idx_out + thread_id + 42*x] = tempOutRow22;
    output[block_start_idx_out + thread_id + 44*x] = tempOutRow23;
    output[block_start_idx_out + thread_id + 46*x] = tempOutRow24;
    output[block_start_idx_out + thread_id + 48*x] = tempOutRow25;
    output[block_start_idx_out + thread_id + 50*x] = tempOutRow26;
    output[block_start_idx_out + thread_id + 52*x] = tempOutRow27;
    output[block_start_idx_out + thread_id + 54*x] = tempOutRow28;
    output[block_start_idx_out + thread_id + 56*x] = tempOutRow29;
    output[block_start_idx_out + thread_id + 58*x] = tempOutRow30;
    output[block_start_idx_out + thread_id + 60*x] = tempOutRow31;
    output[block_start_idx_out + thread_id + 62*x] = tempOutRow32;

    output[block_start_idx_out + thread_id + 64*x] = tempOutRow33;
    output[block_start_idx_out + thread_id + 66*x] = tempOutRow34;
    output[block_start_idx_out + thread_id + 68*x] = tempOutRow35;
    output[block_start_idx_out + thread_id + 70*x] = tempOutRow36;
    output[block_start_idx_out + thread_id + 72*x] = tempOutRow37;
    output[block_start_idx_out + thread_id + 74*x] = tempOutRow38;
    output[block_start_idx_out + thread_id + 76*x] = tempOutRow39;
    output[block_start_idx_out + thread_id + 78*x] = tempOutRow40;
    output[block_start_idx_out + thread_id + 80*x] = tempOutRow41;
    output[block_start_idx_out + thread_id + 82*x] = tempOutRow42;
    output[block_start_idx_out + thread_id + 84*x] = tempOutRow43;
    output[block_start_idx_out + thread_id + 86*x] = tempOutRow44;
    output[block_start_idx_out + thread_id + 88*x] = tempOutRow45;
    output[block_start_idx_out + thread_id + 90*x] = tempOutRow46;
    output[block_start_idx_out + thread_id + 92*x] = tempOutRow47;
    output[block_start_idx_out + thread_id + 94*x] = tempOutRow48;
    
    output[block_start_idx_out + thread_id + 96*x] = tempOutRow49;
    output[block_start_idx_out + thread_id + 98*x] = tempOutRow50;
    output[block_start_idx_out + thread_id + 100*x] = tempOutRow51;
    output[block_start_idx_out + thread_id + 102*x] = tempOutRow52;
    output[block_start_idx_out + thread_id + 104*x] = tempOutRow53;
    output[block_start_idx_out + thread_id + 106*x] = tempOutRow54;
    output[block_start_idx_out + thread_id + 108*x] = tempOutRow55;
    output[block_start_idx_out + thread_id + 110*x] = tempOutRow56;
    output[block_start_idx_out + thread_id + 112*x] = tempOutRow57;
    output[block_start_idx_out + thread_id + 114*x] = tempOutRow58;
    output[block_start_idx_out + thread_id + 116*x] = tempOutRow59;
    output[block_start_idx_out + thread_id + 118*x] = tempOutRow60;
    output[block_start_idx_out + thread_id + 120*x] = tempOutRow61;
    output[block_start_idx_out + thread_id + 122*x] = tempOutRow62;
    output[block_start_idx_out + thread_id + 124*x] = tempOutRow63;
    output[block_start_idx_out + thread_id + 126*x] = tempOutRow64;
}

void decodeImage(float *inputImageR, float *inputImageG, float *inputImageB, char *fileName){
    int index = 0;
    float *tmpBuffer;
    // READ IMAGE and Init buffers
    // const char *fileName = "inputs/2048x2048.jpg";
    Mat fullImage, windowImage;
    Mat channels[3];
    std::vector<float> array;
    fullImage = imread(fileName);
    int imageRows = (int)fullImage.rows, imageCols = (int)fullImage.cols;
    
    split(fullImage, channels);
    array.assign(channels[0].datastart, channels[0].dataend);
    tmpBuffer = &array[0];
    memcpy(inputImageB, tmpBuffer, imageCols * imageRows * sizeof(float));
    array.assign(channels[1].datastart, channels[1].dataend);
    tmpBuffer = &array[0];
    memcpy(inputImageG, tmpBuffer, imageCols * imageRows * sizeof(float));
    array.assign(channels[2].datastart, channels[2].dataend);
    tmpBuffer = &array[0];
    memcpy(inputImageR, tmpBuffer, imageCols * imageRows * sizeof(float));
}

void encodeImage(float *outputR, float *outputG, float *outputB){
    const char *fileName = "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/results/128x128-bl-gpu.jpg";
    vector<Mat> channels;
    Mat finalImage;
    cv::Mat matR = cv::Mat(INPUTHEIGHT*2, INPUTWIDTH*2, CV_32F, outputR);
    cv::Mat matG = cv::Mat(INPUTHEIGHT*2, INPUTWIDTH*2, CV_32F, outputG);
    cv::Mat matB = cv::Mat(INPUTHEIGHT*2, INPUTWIDTH*2, CV_32F, outputB);

    channels.push_back(matB);
    channels.push_back(matG);
    channels.push_back(matR);

    merge(channels, finalImage);
    imwrite(fileName, finalImage); 
}

int main(int argc, char **argv){
    // read in input 2x2 pixels
    char *fileName = argv[1];
    INPUTHEIGHT = atoi(argv[3]);
    INPUTWIDTH = atoi(argv[2]);
    float *inputImageR = (float *)calloc(INPUTWIDTH * INPUTHEIGHT, sizeof(float));
    float *inputImageG = (float *)calloc(INPUTWIDTH * INPUTHEIGHT, sizeof(float)); 
    float *inputImageB = (float *)calloc(INPUTWIDTH * INPUTHEIGHT, sizeof(float));

    decodeImage(inputImageR, inputImageG, inputImageB, fileName);

    float *dev_in_red, *dev_out_red, *dev_in_green, *dev_out_green, *dev_in_blue, *dev_out_blue;
    float *host_out_red, *host_out_green, *host_out_blue;

    host_out_red = (float*) malloc(INPUTHEIGHT * INPUTWIDTH * 4 * sizeof(float));
    host_out_green = (float*) malloc(INPUTHEIGHT * INPUTWIDTH * 4 * sizeof(float));
    host_out_blue = (float*) malloc(INPUTHEIGHT * INPUTWIDTH * 4 * sizeof(float));

    cudaError_t err = cudaMalloc(&dev_in_red, INPUTHEIGHT * INPUTWIDTH *sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }

    err = cudaMalloc(&dev_in_green, INPUTHEIGHT * INPUTWIDTH *sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }

    err = cudaMalloc(&dev_in_blue, INPUTHEIGHT * INPUTWIDTH *sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }

    err = cudaMalloc(&dev_out_red, INPUTHEIGHT * INPUTWIDTH * 4 *sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }

    err = cudaMalloc(&dev_out_green, INPUTHEIGHT * INPUTWIDTH * 4 *sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }

    err = cudaMalloc(&dev_out_blue, INPUTHEIGHT * INPUTWIDTH * 4 *sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }
    
    cudaMemcpy(dev_in_red, inputImageR, INPUTHEIGHT * INPUTWIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_in_green, inputImageG, INPUTHEIGHT * INPUTWIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_in_blue, inputImageB, INPUTHEIGHT * INPUTWIDTH * sizeof(float), cudaMemcpyHostToDevice);

    //create GPU timing events for timing the GPU
    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);        
  float milliseconds = 9999, millisecondsI = 0;
    int block_num = (INPUTHEIGHT * INPUTWIDTH) / (32 * 128);
  for(int l = 0; l < 10; l++){
    cudaEventRecord(st2);
    base_kernel<<<block_num, 256>>>(dev_in_red, dev_out_red, INPUTWIDTH, INPUTHEIGHT);
    base_kernel<<<block_num, 256>>>(dev_in_green, dev_out_green, INPUTWIDTH, INPUTHEIGHT);
    base_kernel<<<block_num, 256>>>(dev_in_blue, dev_out_blue, INPUTWIDTH, INPUTHEIGHT);
    cudaEventRecord(et2);
        
    //host waits until et2 has occured     
    cudaEventSynchronize(et2);
    cudaEventElapsedTime(&millisecondsI, st2, et2);
    if (millisecondsI < milliseconds){
      milliseconds = millisecondsI;
    }
  }

    cout<<INPUTHEIGHT << "x" << INPUTWIDTH << " "<<milliseconds<<"ms"<<endl;

    cudaMemcpy(host_out_red, dev_out_red, INPUTHEIGHT * INPUTWIDTH * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out_green, dev_out_green, INPUTHEIGHT * INPUTWIDTH * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out_blue, dev_out_blue, INPUTHEIGHT * INPUTWIDTH * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // encodeImage(host_out_red, host_out_green, host_out_blue);

    return 0;
}