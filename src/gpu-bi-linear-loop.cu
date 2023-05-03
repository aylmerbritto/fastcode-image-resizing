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
    
    // 2 x 64 -> 4 x 128
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
    for (int i = 0; i < 16; ++i) {
        float top_left = input[block_start_idx + (thread_id / 4) * 2 + (i*2*x)];
        float top_right = input[block_start_idx + (thread_id / 4) * 2 + 1 + (i*2*x)];
        float bottom_left = input[block_start_idx + (thread_id / 4) * 2 + x + (i*2*x)];
        float bottom_right = input[block_start_idx + (thread_id / 4) * 2 + x + 1 + (i*2*x)];

        //__device__ float fmaf ( float  x, float  y, float  z )
        float tempOutRow1 = 0.0f;
        float tempOutRow2 = 0.0f;
        float tempOutRow3 = 0.0f;
        float tempOutRow4 = 0.0f;

        // Top_Left Partial Sums for all rows
        tempOutRow1 = fmaf(top_left, gtop_left_coeff[index], tempOutRow1);
        tempOutRow2 = fmaf(top_left, gtop_left_coeff[index + 4], tempOutRow2);
        tempOutRow3 = fmaf(top_left, gtop_left_coeff[index + 8], tempOutRow3);
        tempOutRow4 = fmaf(top_left, gtop_left_coeff[index + 12], tempOutRow4);

        // Top_Right Partial Sums for all rows
        tempOutRow1 = fmaf(top_right, gtop_right_coeff[index], tempOutRow1);
        tempOutRow2 = fmaf(top_right, gtop_right_coeff[index + 4], tempOutRow2);
        tempOutRow3 = fmaf(top_right, gtop_right_coeff[index + 8], tempOutRow3);
        tempOutRow4 = fmaf(top_right, gtop_right_coeff[index + 12], tempOutRow4);

        // Bottom_Left Partial Sums for all rows
        tempOutRow1 = fmaf(bottom_left, gbottom_left_coeff[index], tempOutRow1);
        tempOutRow2 = fmaf(bottom_left, gbottom_left_coeff[index + 4], tempOutRow2);
        tempOutRow3 = fmaf(bottom_left, gbottom_left_coeff[index + 8], tempOutRow3);
        tempOutRow4 = fmaf(bottom_left, gbottom_left_coeff[index + 12], tempOutRow4);

        // Bottom_Right Partial Sums for all rows
        tempOutRow1 = fmaf(bottom_right, gbottom_right_coeff[index], tempOutRow1);
        tempOutRow2 = fmaf(bottom_right, gbottom_right_coeff[index + 4], tempOutRow2);
        tempOutRow3 = fmaf(bottom_right, gbottom_right_coeff[index + 8], tempOutRow3);
        tempOutRow4 = fmaf(bottom_right, gbottom_right_coeff[index + 12], tempOutRow4);

        output[block_start_idx_out + thread_id + (i * 4 * 2 * x)] = tempOutRow1;
        output[block_start_idx_out + thread_id + 2*x + (i * 4 * 2 * x)] = tempOutRow2;
        output[block_start_idx_out + thread_id + 4*x + (i * 4 * 2 * x)] = tempOutRow3;
        output[block_start_idx_out + thread_id + 6*x + (i * 4 * 2 * x)] = tempOutRow4;
    }
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