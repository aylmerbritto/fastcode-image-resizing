#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <x86intrin.h>
#include "xmmintrin.h"
#include "immintrin.h"

using namespace cv;
using namespace std;

#define MAX_FREQ 3.2
#define BASE_FREQ 2.4

#define WINDOWSIZE 2

// timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc"
                         : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int decodeImage(float *inputImageR, float *inputImageG, float *inputImageB)
{
    int i, j, index = 0;
    float *tmpBuffer;
    // READ IMAGE and Init buffers
    const char *fileName = "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/640x480.jpg";
    Mat fullImage, windowImage;
    Mat channels[3];
    std::vector<float> array;
    fullImage = imread(fileName);
    int imageRows = (int)fullImage.rows, imageCols = (int)fullImage.cols;
    cout << "Width : " << imageCols << endl;
    cout << "Height: " << imageRows << endl;
    // inputImageR = (float *)calloc(fullImage.cols * fullImage.rows, sizeof(float));
    // inputImageG = (float *)calloc(fullImage.cols * fullImage.rows, sizeof(float));
    // inputImageB = (float *)calloc(fullImage.cols * fullImage.rows, sizeof(float));

    for (i = 0; i + WINDOWSIZE <= imageRows; i = i + WINDOWSIZE)
    {
        for (j = 0; j + WINDOWSIZE <= imageCols; j = j + WINDOWSIZE)
        {
            windowImage = fullImage(Range(i, i + WINDOWSIZE), Range(j, j + WINDOWSIZE));
            split(windowImage, channels);
            array.assign(channels[0].datastart, channels[0].dataend);
            tmpBuffer = &array[0];
            memcpy(inputImageB + index, tmpBuffer, 4 * sizeof(float));
            array.assign(channels[1].datastart, channels[1].dataend);
            tmpBuffer = &array[0];
            memcpy(inputImageG + index, tmpBuffer, 4 * sizeof(float));
            array.assign(channels[2].datastart, channels[2].dataend);
            tmpBuffer = &array[0];
            memcpy(inputImageR + index, tmpBuffer, 4 * sizeof(float));
            index = index + 4;
        }
    }
    return 0;
}

int encodeImage(float *outputR, float *outputG, float *outputB){
    const char *fileName = "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/results/640x480.jpg";
    vector<Mat> channels;
    Mat finalImage;
    cv::Mat matR = cv::Mat(480*2, 640*2, CV_32F, outputR);
    cv::Mat matG = cv::Mat(480*2, 640*2, CV_32F, outputG);
    cv::Mat matB = cv::Mat(480*2, 640*2, CV_32F, outputB);

    channels.push_back(matB);
    channels.push_back(matG);
    channels.push_back(matR);

    merge(channels, finalImage);
    imwrite(fileName, finalImage); 
}

void kernel(float *intensityRin, float *intensityGin, float *intensityBin, float *intensityRout, float *intensityGout, float *intensityBout, int rowSize){
    __m128 inR, inG, inB;
    __m128 outRA, outRB, outGA, outGB, outBA, outBB;
    const int mask0 = (0) | (0 << 2) | (1 << 4) | (1 << 6);
    const int mask1 = (2) | (2 << 2) | (3 << 4) | (3 << 6);

    inR = _mm_load_ps(intensityRin);
    inG = _mm_load_ps(intensityGin);
    inB = _mm_load_ps(intensityBin);

    outRA = _mm_permute_ps(inR, mask0);
    outRB = _mm_permute_ps(inR, mask1);
    outGA = _mm_permute_ps(inG, mask0);
    outGB = _mm_permute_ps(inG, mask1);
    outBA = _mm_permute_ps(inB, mask0);
    outBB = _mm_permute_ps(inB, mask1);

    _mm_store_ps(intensityRout + (0 * rowSize), outRA);
    _mm_store_ps(intensityRout + (1 * rowSize), outRA);
    _mm_store_ps(intensityGout + (0 * rowSize), outGA);
    _mm_store_ps(intensityGout + (1 * rowSize), outGA);
    _mm_store_ps(intensityBout + (0 * rowSize), outBA);
    _mm_store_ps(intensityBout + (1 * rowSize), outBA);
    
    _mm_store_ps(intensityRout + (2 * rowSize), outRB);
    _mm_store_ps(intensityRout + (3 * rowSize), outRB);
    _mm_store_ps(intensityGout + (2 * rowSize), outGB);
    _mm_store_ps(intensityGout + (3 * rowSize), outGB);
    _mm_store_ps(intensityBout + (2 * rowSize), outBB);
    _mm_store_ps(intensityBout + (3 * rowSize), outBB);
}

int main(int argc, char **argv)
{
    unsigned long long t0, t1;
    // kernel width
    int outputRowSize = 1280;
    int outputColumnSize = 960;
    int inputIndex, outputRow, outputColumn, outputIndex;

    // Output Image Stack defintion
    float *outputR = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    float *outputG = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    float *outputB = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    
    // float inputImageR[12] = {1, 5, 6, 7, 80, 100, 150, 255,1, 5, 6, 7};
    // float inputImageG[12] = {80, 100, 150, 255, 1, 5, 6, 7,6, 7, 80, 100};
    // float inputImageB[12] = {123, 154, 112, 111, 80, 100, 150, 255,1, 5, 6, 7};
    float *inputImageR = (float *)calloc(640 * 480, sizeof(float));
    float *inputImageG = (float *)calloc(640 * 480, sizeof(float));
    float *inputImageB = (float *)calloc(640 * 480, sizeof(float));
    decodeImage(inputImageR, inputImageG, inputImageB);
    t0 = rdtsc();
    for(int i = 0; i < (outputColumnSize*outputRowSize)/16 ; i++){
        inputIndex = (i*4);
        outputRow = 4*((i*2)/(outputRowSize/2));
        outputColumn = 2*((i*2)%(outputRowSize/2));
        outputIndex = (outputRow*outputRowSize)+outputColumn;
        // cout << i << '\t' << inputImageR[inputIndex] << endl;
        kernel(inputImageR+inputIndex, inputImageG+inputIndex, inputImageB+inputIndex,outputR+outputIndex, outputG+outputIndex, outputB+outputIndex, outputRowSize);
    }
    t1 = rdtsc();
    printf("cycles taken (I think): %f\n", ((double)(t1 - t0) * MAX_FREQ / BASE_FREQ));
    encodeImage(outputR, outputG, outputB);
    // cout << "R=========================\n";
    // for (int i = 0; i < 16;i++)
    // {
    //     for (int j = 0; j < 16; j++)
    //     {
    //         cout << outputR[(i*outputRowSize)+j] << '\t';
    //     }
    //     cout << '\n';
    // }
    // cout << "G=========================\n";
    // for (int i = 0; i < outputColumnSize*outputRowSize;)
    // {
    //     for (int j = 0; j < outputRowSize; j++)
    //     {
    //         cout << outputG[i++] << '\t';
    //     }
    //     cout << '\n';
    // }
    // cout << "B=========================\n";
    // for (int i = 0; i < outputColumnSize*outputRowSize;)
    // {
    //     for (int j = 0; j < outputRowSize; j++)
    //     {
    //         cout << outputB[i++] << '\t';
    //     }
    //     cout << '\n';
    // }
    free(outputR);
    free(outputG);
    free(outputB);
}