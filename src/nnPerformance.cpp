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

#define INPUTWIDTH 640
#define INPUTHEIGHT 480

#define NUMBER_OF_RUNS 100

// timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc"
                         : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int decodeImage(float *inputImageR, float *inputImageG, float *inputImageB,char *fileName)
{
    int i, j, index = 0;
    float *tmpBuffer;
    // const char *fileName = "inputs/640x480.jpg";
    // READ IMAGE and Init buffers
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
    return 0;
}


int encodeImage(float *outputR, float *outputG, float *outputB){
    const char *fileName = "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/results/640x480-bl.jpg";
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

void kernel(float *intensityRin, float *intensityGin, float *intensityBin, float *intensityRout, float *intensityGout, float *intensityBout, int rowSize)
{
    __m128 inR, inG, inB;
    __m128 outRA, outRB, outGA, outGB, outBA, outBB;
    int mask0 = (0) | (0 << 2) | (1 << 4) | (1 << 6);
    int mask1 = (2) | (2 << 2) | (3 << 4) | (3 << 6);

    
    
    inR = _mm_load_ps(intensityRin);
    outRA = _mm_permute_ps(inR, mask0);
    outRB = _mm_permute_ps(inR, mask1);
    inB = _mm_load_ps(intensityBin);
    outBA = _mm_permute_ps(inB, mask0);
    outBB = _mm_permute_ps(inB, mask1);
    inG = _mm_load_ps(intensityGin);
    outGA = _mm_permute_ps(inG, mask0);
    outGB = _mm_permute_ps(inG, mask1);

    _mm_store_ps(intensityRout + (0 * rowSize), outRA);
    _mm_store_ps(intensityRout + (1 * rowSize), outRA);
    
    _mm_store_ps(intensityRout + 4, outRB);
    _mm_store_ps(intensityRout + 4 + rowSize, outRB);
    

    
    _mm_store_ps(intensityBout + (0 * rowSize), outBA);
    _mm_store_ps(intensityBout + (1 * rowSize), outBA);
    
    _mm_store_ps(intensityBout + 4, outBB);
    _mm_store_ps(intensityBout + (4+ rowSize), outBB);
    
    
    _mm_store_ps(intensityGout + (0 * rowSize), outGA);
    _mm_store_ps(intensityGout + (1 * rowSize), outGA);
    
    _mm_store_ps(intensityGout + 4, outGB);
    _mm_store_ps(intensityGout + (4+ rowSize), outGB);
}



int main(int argc, char **argv)
{
    char *fileName = argv[1];
    int gnWidth = atoi(argv[2]);
    int gnHeight = atoi(argv[3]);
    unsigned long long t0, t1;
    long double sum=0, packingTime = 0, GFLOPS=0;
    // kernel width
    int outputRowSize = gnWidth*2;
    int outputColumnSize = gnHeight*2;
    int inputIndex, outputRow, outputColumn, outputIndex, OoutputRow, OoutputColumn, OoutputIndex, OinputIndex;

    // Output Image Stack defintion
    float *outputR = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    float *outputG = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    float *outputB = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));

    // read in input 2x2 pixels
    float *inputImageR = (float *)calloc(gnHeight * gnWidth, sizeof(float));
    float *inputImageG = (float *)calloc(gnHeight * gnWidth, sizeof(float)); 
    float *inputImageB = (float *)calloc(gnHeight * gnWidth, sizeof(float));
    t0 = rdtsc();
    decodeImage(inputImageR, inputImageG, inputImageB, fileName);
    t1 = rdtsc();
    packingTime = (t1-t0) * MAX_FREQ / BASE_FREQ;
    float *coefficients = (float *)calloc(4 * 4 * 4, sizeof(float));
    
    OoutputIndex = 0; OinputIndex = 0;
    double localSum = 0;
    // long double minTime = 4000000000000000;
    // for(int t=0;t<(gnHeight*gnWidth/256);t++){
    //     OoutputRow = 16*((t*16)/(outputRowSize/2));
    //     OoutputColumn = 2*((t*16)%(outputRowSize/2));
    //     OoutputIndex = (OoutputRow*outputRowSize)+OoutputColumn;
    //     OinputIndex = ((OoutputRow*outputRowSize)/4)+(outputColumn/2);
    //     if(gnHeight>16){
    //         outputColumnSize = 32;
    //         outputRowSize = 32;
    //     }
        long double minTime = 4000000000;
        for(int k = 0; k < NUMBER_OF_RUNS; k++){
            sum=0;
            for(int i = 0; i < (outputColumnSize*outputRowSize)/16 ; i++){
                outputRow = 4*((i*2)/(outputRowSize/2));
                outputColumn = 2*((i*2)%(outputRowSize/2));
                outputIndex = (outputRow*outputRowSize)+outputColumn;
                inputIndex = ((outputRow*outputRowSize)/4)+(outputColumn/2);
                t0 = rdtsc();
                kernel(inputImageR + inputIndex, inputImageG + inputIndex, inputImageB + inputIndex, outputR + outputIndex, outputG + outputIndex, outputB + outputIndex, outputRowSize);
                t1 = rdtsc();
                sum=sum+(t1-t0);
            }
            if(sum<minTime){
                minTime =sum;
            }
        // }
        // localSum = localSum+minTime;
    }
    // sum = ((sum) * MAX_FREQ / BASE_FREQ)/NUMBER_OF_RUNS;
    sum = minTime* MAX_FREQ / BASE_FREQ;
    GFLOPS = (12*((gnHeight*gnWidth*4)/16))/sum;
    cout << gnHeight <<','<< GFLOPS <<','<< sum<<',';
    free(coefficients);
    free(outputR);
    free(outputG);
    free(outputB);
}
