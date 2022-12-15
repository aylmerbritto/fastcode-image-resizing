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
    const char *fileName = "inputs/8x8.jpg";
    Mat fullImage, windowImage;
    Mat channels[3];
    std::vector<float> array;
    fullImage = imread(fileName);
    int imageRows = (int)fullImage.rows, imageCols = (int)fullImage.cols;
    cout << "Width : " << imageCols << endl;
    cout << "Height: " << imageRows << endl;
    
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
    cout << "RGB Channels Read and assgined"<<endl;
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


void generateCoefficients(float *coefficients)
{
    /*
        TODO: To calculate Z,A,B,C,D
        The Values are going to be the same throughout given the kernel input and output are the same
    */
    // float Z = 1 / 9;
    // int kernelSauceIndex = 0, i;
    // float a1[4] = {3, 3, 3, 3};
    // float a2[4] = {2, 2, 2, 2};
    // float a3[4] = {1, 1, 1, 1};
    // float a4[4] = {0, 0, 0, 0};
    // float b4[4] = {3, 3, 3, 3};
    // float b3[4] = {2, 2, 2, 2};
    // float b2[4] = {1, 1, 1, 1};
    // float b1[4] = {0, 0, 0, 0};
    // float c[4] = {3, 2, 1, 0};
    // float d[4] = {0, 1, 2, 3};
    __m128 vec1, vec2, vec3, vec4;
    // float *result = (float*)calloc(4,sizeof(float));
    /*
        a1*c, a2*c, a3*c, a4*c
        a1*d, a2*d, a3*d, a4*d
        b1*c, b2*c, b3*c, b4*c
        b1*d, b2*d, b3*d, b4*d
    */

    float vec1flt[4] = {1, 2 / 3.0, 1 / 3.0, 0};
    float vec2flt[4] = {6 / 9.0, 4 / 9.0, 2 / 9.0, 0};
    float vec3flt[4] = {3 / 9.0, 2 / 9.0, 1 / 9.0, 0};
    float vec4flt[4] = {0, 0, 0, 0};
    vec1 = _mm_load_ps(vec1flt);
    vec2 = _mm_load_ps(vec2flt);
    vec3 = _mm_load_ps(vec3flt);
    vec4 = _mm_load_ps(vec4flt);
    _mm_store_ps(coefficients, vec1);
    _mm_store_ps(coefficients + 4, vec2);
    _mm_store_ps(coefficients + 8, vec3);
    _mm_store_ps(coefficients + 12, vec4);

    float vec5flt[4] = {0, 1 / 3.0, 2 / 3.0, 1};
    float vec6flt[4] = {0, 2 / 9.0, 4 / 9.0, 6 / 9.0};
    float vec7flt[4] = {0, 1 / 9.0, 2 / 9.0, 3 / 9.0};
    float vec8flt[4] = {0, 0, 0, 0};
    vec1 = _mm_load_ps(vec5flt);
    vec2 = _mm_load_ps(vec6flt);
    vec3 = _mm_load_ps(vec7flt);
    vec4 = _mm_load_ps(vec8flt);  
    _mm_store_ps(coefficients + 16, vec1);
    _mm_store_ps(coefficients + 20, vec2);
    _mm_store_ps(coefficients + 24, vec3);
    _mm_store_ps(coefficients + 28, vec4);

    float vec9flt[4] = {0, 0, 0, 0};
    float vec10flt[4] = {3 / 9.0, 2 / 9.0, 1 / 9.0, 0};
    float vec11flt[4] = {6 / 9.0, 4 / 9.0, 2 / 9.0, 0};
    float vec12flt[4] = {1, 2 / 3.0, 1 / 3.0, 0};
    vec1 = _mm_load_ps(vec9flt);
    vec2 = _mm_load_ps(vec10flt);
    vec3 = _mm_load_ps(vec11flt);
    vec4 = _mm_load_ps(vec12flt);   
    _mm_store_ps(coefficients + 32, vec1);
    _mm_store_ps(coefficients + 36, vec2);
    _mm_store_ps(coefficients + 40, vec3);
    _mm_store_ps(coefficients + 44, vec4);

    float vec13flt[4] = {0, 0, 0, 0};
    float vec14flt[4] = {0, 1 / 9.0, 2 / 9.0, 3 / 9.0};
    float vec15flt[4] = {0, 2 / 9.0, 4 / 9.0, 6 / 9.0};
    float vec16flt[4] = {0, 1 / 3.0, 2 / 3.0, 1.0};
    vec1 = _mm_load_ps(vec13flt);
    vec2 = _mm_load_ps(vec14flt);
    vec3 = _mm_load_ps(vec15flt);
    vec4 = _mm_load_ps(vec16flt);

    _mm_store_ps(coefficients + 48, vec1);
    _mm_store_ps(coefficients + 52, vec2);
    _mm_store_ps(coefficients + 56, vec3);
    _mm_store_ps(coefficients + 60, vec4);
}

void kernel(float *intensityRin, float *intensityGin, float *intensityBin, float *intensityRout, float *intensityGout, float *intensityBout, float *coefficients, int rowSize)
{
    float *address;
    // load inR, inG, inB pixel values (4 values each)

    //Initializing the first row of output Pixels
    __m128 outRA = _mm_setzero_ps(); __m128 outGA = _mm_setzero_ps(); __m128 outBA = _mm_setzero_ps();
    // Q11 over the SIMD Register
    __m128 tmpR = _mm_broadcast_ss(intensityRin); __m128 tmpG = _mm_broadcast_ss(intensityGin); __m128 tmpB = _mm_broadcast_ss(intensityBin);
    //Load A1*C*Z
    __m128 coefsA = _mm_load_ps(coefficients + 0); // 
    outRA = _mm_fmadd_ps(tmpR, coefsA, outRA); outGA = _mm_fmadd_ps(tmpG, coefsA, outGA); outBA = _mm_fmadd_ps(tmpB, coefsA, outBA);
    // Q12 over the SIMD Register
    tmpR = _mm_broadcast_ss(intensityRin+1); tmpG = _mm_broadcast_ss(intensityGin+1); tmpB = _mm_broadcast_ss(intensityBin+1);
    //Load A2*C*Z
    __m128 coefsB = _mm_load_ps(coefficients + 4);
    __m128 outRB = _mm_setzero_ps(); __m128 outGB = _mm_setzero_ps(); __m128 outBB = _mm_setzero_ps(); 
    outRB = _mm_fmadd_ps(tmpR, coefsB, outRB); outGB = _mm_fmadd_ps(tmpG, coefsB, outGB); outBB = _mm_fmadd_ps(tmpB, coefsB, outBB);

    coefsA = _mm_load_ps(coefficients + 16);
    outRA = _mm_fmadd_ps(tmpR, coefsA, outRA); outGA = _mm_fmadd_ps(tmpG, coefsA, outGA); outBA = _mm_fmadd_ps(tmpB, coefsA, outBA);
    coefsB = _mm_load_ps(coefficients + 20);
    outRB = _mm_fmadd_ps(tmpR, coefsB, outRB); outGB = _mm_fmadd_ps(tmpG, coefsB, outGB); outBB = _mm_fmadd_ps(tmpB, coefsB, outBB);

    // Q21 over the SIMD Register
    tmpR = _mm_broadcast_ss(intensityRin+(rowSize/2)); tmpG = _mm_broadcast_ss(intensityGin+(rowSize/2)); tmpB = _mm_broadcast_ss(intensityBin+(rowSize/2));
    coefsA = _mm_load_ps(coefficients + 32);
    outRA = _mm_fmadd_ps(tmpR, coefsA, outRA); outGA = _mm_fmadd_ps(tmpG, coefsA, outGA); outBA = _mm_fmadd_ps(tmpB, coefsA, outBA);
    coefsB = _mm_load_ps(coefficients + 36);
    outRB = _mm_fmadd_ps(tmpR, coefsB, outRB); outGB = _mm_fmadd_ps(tmpG, coefsB, outGB); outBB = _mm_fmadd_ps(tmpB, coefsB, outBB);
    
    tmpR = _mm_broadcast_ss(intensityRin+(rowSize/2)+1); tmpG = _mm_broadcast_ss(intensityGin+(rowSize/2)+1); tmpB = _mm_broadcast_ss(intensityBin+(rowSize/2)+1);
    coefsA = _mm_load_ps(coefficients + 48);
    outRA = _mm_fmadd_ps(tmpR, coefsA, outRA); outGA = _mm_fmadd_ps(tmpG, coefsA, outGA); outBA = _mm_fmadd_ps(tmpB, coefsA, outBA);
    coefsB = _mm_load_ps(coefficients + 52);
    outRB = _mm_fmadd_ps(tmpR, coefsB, outRB); outGB = _mm_fmadd_ps(tmpG, coefsB, outGB); outBB = _mm_fmadd_ps(tmpB, coefsB, outBB);

    _mm_store_ps(intensityRout + (0 * rowSize), outRA); _mm_store_ps(intensityGout + (0 * rowSize), outGA); _mm_store_ps(intensityBout + (0 * rowSize), outBA);
    _mm_store_ps(intensityRout + (1 * rowSize), outRB); _mm_store_ps(intensityGout + (1 * rowSize), outGB); _mm_store_ps(intensityBout + (1 * rowSize), outBB);
    
    outRA = _mm_setzero_ps(); outGA = _mm_setzero_ps(); outBA = _mm_setzero_ps();
    coefsB = _mm_load_ps(coefficients + 12);
    tmpR = _mm_broadcast_ss(intensityRin); tmpG = _mm_broadcast_ss(intensityGin); tmpB = _mm_broadcast_ss(intensityBin);
    coefsA = _mm_load_ps(coefficients + 8);
    outRA = _mm_fmadd_ps(tmpR, coefsA, outRA); outGA = _mm_fmadd_ps(tmpG, coefsA, outGA); outBA = _mm_fmadd_ps(tmpB, coefsA, outBA);

    outRB = _mm_setzero_ps(); outGB = _mm_setzero_ps(); outBB = _mm_setzero_ps(); 
    outRB = _mm_fmadd_ps(tmpR, coefsB, outRB); outGB = _mm_fmadd_ps(tmpG, coefsB, outGB); outBB = _mm_fmadd_ps(tmpB, coefsB, outBB);
    coefsA = _mm_load_ps(coefficients + 24);
    tmpR = _mm_broadcast_ss(intensityRin+1); tmpG = _mm_broadcast_ss(intensityGin+1); tmpB = _mm_broadcast_ss(intensityBin+1);
    outRA = _mm_fmadd_ps(tmpR, coefsA, outRA); outGA = _mm_fmadd_ps(tmpG, coefsA, outGA);outBA = _mm_fmadd_ps(tmpB, coefsA, outBA);
    coefsB = _mm_load_ps(coefficients + 28);
    outRB = _mm_fmadd_ps(tmpR, coefsB, outRB); outGB = _mm_fmadd_ps(tmpG, coefsB, outGB);outBB = _mm_fmadd_ps(tmpB, coefsB, outBB);
    
    tmpR = _mm_broadcast_ss(intensityRin+(rowSize/2)); tmpG = _mm_broadcast_ss(intensityGin+(rowSize/2)); tmpB = _mm_broadcast_ss(intensityBin+(rowSize/2));
    coefsA = _mm_load_ps(coefficients + 40);
    outRA = _mm_fmadd_ps(tmpR, coefsA, outRA); outGA = _mm_fmadd_ps(tmpG, coefsA, outGA); outBA = _mm_fmadd_ps(tmpB, coefsA, outBA);
    coefsB = _mm_load_ps(coefficients + 44);
    outRB = _mm_fmadd_ps(tmpR, coefsB, outRB); outGB = _mm_fmadd_ps(tmpG, coefsB, outGB); outBB = _mm_fmadd_ps(tmpB, coefsB, outBB);

    tmpR = _mm_broadcast_ss(intensityRin+(rowSize/2)+1); tmpG = _mm_broadcast_ss(intensityGin+(rowSize/2)+1); tmpB = _mm_broadcast_ss(intensityBin+(rowSize/2)+1);
    coefsA = _mm_load_ps(coefficients + 56);
    outRA = _mm_fmadd_ps(tmpR, coefsA, outRA); outGA = _mm_fmadd_ps(tmpG, coefsA, outGA); outBA = _mm_fmadd_ps(tmpB, coefsA, outBA);
    coefsB = _mm_load_ps(coefficients + 60);
    outRB = _mm_fmadd_ps(tmpR, coefsB, outRB); outGB = _mm_fmadd_ps(tmpG, coefsB, outGB);outBB = _mm_fmadd_ps(tmpB, coefsB, outBB);

    _mm_store_ps(intensityRout + (2 * rowSize), outRA); _mm_store_ps(intensityGout + (2 * rowSize), outGA); _mm_store_ps(intensityBout + (2 * rowSize), outBA);
    _mm_store_ps(intensityGout + (3 * rowSize), outGB); _mm_store_ps(intensityRout + (3 * rowSize), outRB); _mm_store_ps(intensityBout + (3 * rowSize), outBB);  
}


int main(int argc, char **argv)
{
    unsigned long long t0, t1, sum=0;
    // kernel width
    int outputRowSize = INPUTWIDTH*2;
    int outputColumnSize = INPUTHEIGHT*2;
    int inputIndex, outputRow, outputColumn, outputIndex;

    // Output Image Stack defintion
    float *outputR = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    float *outputG = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    float *outputB = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));

    // read in input 2x2 pixels
    float *inputImageR = (float *)calloc(INPUTWIDTH * INPUTHEIGHT, sizeof(float));
    float *inputImageG = (float *)calloc(INPUTWIDTH * INPUTHEIGHT, sizeof(float)); 
    float *inputImageB = (float *)calloc(INPUTWIDTH * INPUTHEIGHT, sizeof(float));
    t0 = rdtsc();
    decodeImage(inputImageR, inputImageG, inputImageB);
    t1 = rdtsc();
    // printf("PREPROCESSING TIME %f\n", ((double)(t1-t0) * MAX_FREQ / BASE_FREQ));
    float *coefficients = (float *)calloc(4 * 4 * 4, sizeof(float));
    generateCoefficients(coefficients);
    int n16 = INPUTWIDTH/16; 
    // generate coefficients
    for(int i = 0; i < (outputColumnSize*outputRowSize)/16 ; i++){
        outputRow = 4*((i*2)/(outputRowSize/2));
        outputColumn = 2*((i*2)%(outputRowSize/2));
        outputIndex = (outputRow*outputRowSize)+outputColumn;
        inputIndex = ((outputRow*outputRowSize)/4)+(outputColumn/2);
        // cout << i << "  " <<inputIndex<< "  "  << outputIndex<<endl;
        t0 = rdtsc();
        kernel(inputImageR+inputIndex, inputImageG+inputIndex, inputImageB+inputIndex,outputR+outputIndex, outputG+outputIndex, outputB+outputIndex,coefficients, outputRowSize);
        t1 = rdtsc();
        sum=sum+(t1-t0);
    }
    
    sum =  ((sum) * MAX_FREQ / BASE_FREQ);
    double GFLOPS = (2*48*4*((INPUTHEIGHT*INPUTWIDTH*4)/16))/sum;
    cout << GFLOPS << ","<<GFLOPS<< endl;
    encodeImage(outputR, outputG, outputB);
    free(coefficients);
    free(outputR);
    free(outputG);
    free(outputB);
}



/*
1. how many 16x16 images
2. inputR = skip 16x16 = 256
2. outputR  = skip 32x32 = 
*/