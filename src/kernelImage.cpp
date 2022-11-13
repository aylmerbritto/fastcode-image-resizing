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

#define ROWS 4
#define COLS 4

// timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc"
                         : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
char *home = "/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/";

int decodeImage(float *inputImageR, float *inputImageG, float *inputImageB, char *fileName)
{
    int i, j, index = 0;
    float *tmpBuffer;
    // READ IMAGE and Init buffers
    // const char *fileName = "/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/inputs/640x480.jpg";
    Mat fullImage;
    Mat windowImage;
    Mat channels[3];
    std::vector<float> array;
    fullImage = imread(fileName);
    int imageRows = (int)fullImage.rows;
    int imageCols = (int)fullImage.cols;
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

int encodeImage(float *outputR, float *outputG, float *outputB, char *fileName)
{
    // const char *fileName = "/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/results/640x480-bl.jpg";
    vector<Mat> channels;
    Mat finalImage;
    // cv::Mat matR = cv::Mat(480 * 2, 640 * 2, CV_32F, outputR);
    // cv::Mat matG = cv::Mat(480 * 2, 640 * 2, CV_32F, outputG);
    // cv::Mat matB = cv::Mat(480 * 2, 640 * 2, CV_32F, outputB);
    cv::Mat matR = cv::Mat(ROWS * 2, COLS * 2, CV_32F, outputR);
    cv::Mat matG = cv::Mat(ROWS * 2, COLS * 2, CV_32F, outputG);
    cv::Mat matB = cv::Mat(ROWS * 2, COLS * 2, CV_32F, outputB);

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

    __m128 vec1, vec2, vec3, vec4;
    /*
        a1*c, a2*c, a3*c, a4*c
        a1*d, a2*d, a3*d, a4*d
        b1*c, b2*c, b3*c, b4*c
        b1*d, b2*d, b3*d, b4*d
    */

    // float vec1flt[4] = {0.0 / 9.0, 3.0 / 9.0, 6.0 / 9.0, 9.0 / 9.0};
    // float vec2flt[4] = {9.0 / 9.0, 6.0 / 9.0, 3.0 / 3.0, 0.0 / 9.0};
    // float vec3flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec4flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};

    float vec1flt[4] = {9.0 / 9.0, 6.0 / 9.0, 3.0 / 9.0, 0.0 / 9.0};
    float vec2flt[4] = {0.0 / 9.0, 3.0 / 9.0, 6.0 / 3.0, 9.0 / 9.0};
    float vec3flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    float vec4flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec1flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec2flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec3flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec4flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    vec1 = _mm_load_ps(vec1flt);
    vec2 = _mm_load_ps(vec2flt);
    vec3 = _mm_load_ps(vec3flt);
    vec4 = _mm_load_ps(vec4flt);
    // vec1 = _mm_set_ps(0, 1/3.0, 2/3.0, 1); //z*a1*c
    // vec2 = _mm_set_ps(0, 2/9.0, 4/9.0, 6/9.0); //z*a2*c
    // vec3 = _mm_set_ps(0, 1/9.0, 2/9.0, 3/9.0); //z*a3*c
    // vec4 = _mm_set_ps(0, 0, 0, 0); //z*a4*c
    _mm_store_ps(coefficients, vec1);
    _mm_store_ps(coefficients + 4, vec2);
    _mm_store_ps(coefficients + 8, vec3);
    _mm_store_ps(coefficients + 12, vec4);

    float vec5flt[4] = {6.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0, 0.0 / 9.0};
    float vec6flt[4] = {0.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0, 6.0 / 9.0};
    float vec7flt[4] = {3.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0, 0.0 / 9.0};
    float vec8flt[4] = {0.0 / 9.0, 1.0 / 9.0, 2.0 / 9.0, 3.0 / 9.0};
    // float vec5flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec6flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec7flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec8flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    vec1 = _mm_load_ps(vec5flt);
    vec2 = _mm_load_ps(vec6flt);
    vec3 = _mm_load_ps(vec7flt);
    vec4 = _mm_load_ps(vec8flt);
    // vec1 = _mm_set_ps(1, 2 / 3.0, 1 / 3.0, 0);       // z*a1*d
    // vec2 = _mm_set_ps(6 / 9.0, 4 / 9.0, 2 / 9.0, 0); // z*a2*d
    // vec3 = _mm_set_ps(3 / 9.0, 2 / 9.0, 1 / 9.0, 0); // z*a3*d
    // vec4 = _mm_set_ps(0, 0, 0, 0);                   // z*a4*c
    _mm_store_ps(coefficients + 16, vec1);
    _mm_store_ps(coefficients + 20, vec2);
    _mm_store_ps(coefficients + 24, vec3);
    _mm_store_ps(coefficients + 28, vec4);

    float vec9flt[4] = {3.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0, 0.0 / 9.0};
    float vec10flt[4] = {0.0 / 9.0, 1.0 / 9.0, 2.0 / 9.0, 3.0 / 9.0};
    float vec11flt[4] = {6.0 / 9.0, 4.0 / 9.0, 2.0 / 4.0, 0.0 / 9.0};
    float vec12flt[4] = {0.0 / 9.0, 2.0 / 4.0, 4.0 / 9.0, 6.0 / 9.0};
    // float vec9flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec10flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec11flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec12flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    vec1 = _mm_load_ps(vec9flt);
    vec2 = _mm_load_ps(vec10flt);
    vec3 = _mm_load_ps(vec11flt);
    vec4 = _mm_load_ps(vec12flt);
    // vec4 = _mm_set_ps(0, 1 / 3.0, 2 / 3.0, 1);       // z*b4*c
    // vec3 = _mm_set_ps(0, 2 / 9.0, 4 / 9.0, 6 / 9.0); // z*b3*c
    // vec2 = _mm_set_ps(0, 1 / 9.0, 2 / 9.0, 3 / 9.0); // z*b2*c
    // vec1 = _mm_set_ps(0, 0, 0, 0);                   // z*b1*c
    _mm_store_ps(coefficients + 32, vec1);
    _mm_store_ps(coefficients + 36, vec2);
    _mm_store_ps(coefficients + 40, vec3);
    _mm_store_ps(coefficients + 44, vec4);

    float vec13flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    float vec14flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    float vec15flt[4] = {9.0 / 9.0, 6.0 / 9.0, 3.0 / 9.0, 0.0 / 9.0};
    float vec16flt[4] = {0.0 / 9.0, 3.0 / 9.0, 6.0 / 9.0, 9.0 / 9.0};
    // float vec13flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec14flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec15flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    // float vec16flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    vec1 = _mm_load_ps(vec13flt);
    vec2 = _mm_load_ps(vec14flt);
    vec3 = _mm_load_ps(vec15flt);
    vec4 = _mm_load_ps(vec16flt);

    // vec4 = _mm_set_ps(1.0, 2 / 3.0, 1 / 3.0, 0.0);     // z*b4*d
    // vec3 = _mm_set_ps(6 / 9.0, 4 / 9.0, 2 / 9.0, 0.0); // z*b3*d
    // vec2 = _mm_set_ps(3 / 9.0, 2 / 9.0, 1 / 9.0, 0.0); // z*b2*d
    // vec1 = _mm_set_ps(0.0, 0.0, 0.0, 0.0);             // z*b1*d
    _mm_store_ps(coefficients + 48, vec1);
    _mm_store_ps(coefficients + 52, vec2);
    _mm_store_ps(coefficients + 56, vec3);
    _mm_store_ps(coefficients + 60, vec4);
}

void kernel(float *intensityRin, float *intensityGin, float *intensityBin, float *intensityRout, float *intensityGout, float *intensityBout, int row_i, int col_i, float *coefficients, int rowSize, int colSize)
{

    __m128 RQ11, RQ21, RQ12, RQ22;
    __m128 GQ11, GQ21, GQ12, GQ22;
    __m128 BQ11, BQ21, BQ12, BQ22;

    __m128 Rrow0, Rrow1, Rrow2, Rrow3;
    __m128 Grow0, Grow1, Grow2, Grow3;
    __m128 Brow0, Brow1, Brow2, Brow3;

    __m128 AC, AD, BC, BD;

    // same for all pixels in matrix
    // RQ11 = _mm_broadcast_ss(&intensityRin[8 * row_i + 4 * col_i + 0]);
    // RQ21 = _mm_broadcast_ss(&intensityRin[8 * row_i + 4 * col_i + 2]);
    // RQ12 = _mm_broadcast_ss(&intensityRin[8 * row_i + 4 * col_i + 1]);
    // RQ22 = _mm_broadcast_ss(&intensityRin[8 * row_i + 4 * col_i + 3]);

    // GQ11 = _mm_broadcast_ss(&intensityGin[8 * row_i + 4 * col_i + 0]);
    // GQ21 = _mm_broadcast_ss(&intensityGin[8 * row_i + 4 * col_i + 2]);
    // GQ12 = _mm_broadcast_ss(&intensityGin[8 * row_i + 4 * col_i + 1]);
    // GQ22 = _mm_broadcast_ss(&intensityGin[8 * row_i + 4 * col_i + 3]);

    // BQ11 = _mm_broadcast_ss(&intensityBin[8 * row_i + 4 * col_i + 0]);
    // BQ21 = _mm_broadcast_ss(&intensityBin[8 * row_i + 4 * col_i + 2]);
    // BQ12 = _mm_broadcast_ss(&intensityBin[8 * row_i + 4 * col_i + 1]);
    // BQ22 = _mm_broadcast_ss(&intensityBin[8 * row_i + 4 * col_i + 3]);
    // cout << "r vals: " << intensityRin[4 * row_i + 8 * col_i + 0] << "\t" << intensityRin[4 * row_i + 8 * col_i + 1] << "\t" << intensityRin[4 * row_i + 8 * col_i + 2] << "\t" << intensityRin[4 * row_i + 8 * col_i + 3] << endl;
    RQ11 = _mm_broadcast_ss(&intensityRin[4 * row_i + 4 * col_i + 0]);
    RQ21 = _mm_broadcast_ss(&intensityRin[4 * row_i + 4 * col_i + 2]);
    RQ12 = _mm_broadcast_ss(&intensityRin[4 * row_i + 4 * col_i + 1]);
    RQ22 = _mm_broadcast_ss(&intensityRin[4 * row_i + 4 * col_i + 3]);

    GQ11 = _mm_broadcast_ss(&intensityGin[4 * row_i + 4 * col_i + 0]);
    GQ21 = _mm_broadcast_ss(&intensityGin[4 * row_i + 4 * col_i + 2]);
    GQ12 = _mm_broadcast_ss(&intensityGin[4 * row_i + 4 * col_i + 1]);
    GQ22 = _mm_broadcast_ss(&intensityGin[4 * row_i + 4 * col_i + 3]);

    BQ11 = _mm_broadcast_ss(&intensityBin[4 * row_i + 4 * col_i + 0]);
    BQ21 = _mm_broadcast_ss(&intensityBin[4 * row_i + 4 * col_i + 2]);
    BQ12 = _mm_broadcast_ss(&intensityBin[4 * row_i + 4 * col_i + 1]);
    BQ22 = _mm_broadcast_ss(&intensityBin[4 * row_i + 4 * col_i + 3]);

    // these all start out to be 0
    // cout << "row 0: " << (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 0 * rowSize) << endl;
    // cout << "row 1: " << (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 1 * rowSize) << endl;
    // cout << "row 2: " << (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 2 * rowSize) << endl;
    // cout << "row 3: " << (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 3 * rowSize) << endl;
    cout << endl;
    Rrow0 = _mm_load_ps(intensityRout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 0 * rowSize));
    Grow0 = _mm_load_ps(intensityGout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 0 * rowSize));
    Brow0 = _mm_load_ps(intensityBout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 0 * rowSize));

    Rrow1 = _mm_load_ps(intensityRout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 1 * rowSize));
    Grow1 = _mm_load_ps(intensityGout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 1 * rowSize));
    Brow1 = _mm_load_ps(intensityBout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 1 * rowSize));

    Rrow2 = _mm_load_ps(intensityRout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 2 * rowSize));
    Grow2 = _mm_load_ps(intensityGout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 2 * rowSize));
    Brow2 = _mm_load_ps(intensityBout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 2 * rowSize));

    Rrow3 = _mm_load_ps(intensityRout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 3 * rowSize));
    Grow3 = _mm_load_ps(intensityGout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 3 * rowSize));
    Brow3 = _mm_load_ps(intensityBout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 3 * rowSize));

    // load in coefficients for row 0
    AC = _mm_load_ps(coefficients + 0);
    AD = _mm_load_ps(coefficients + 4);
    BC = _mm_load_ps(coefficients + 8);
    BD = _mm_load_ps(coefficients + 12);

    // multiply Q11 with AC
    Rrow0 = _mm_fmadd_ps(RQ11, AC, Rrow0);
    Rrow0 = _mm_fmadd_ps(RQ21, AD, Rrow0);
    Rrow0 = _mm_fmadd_ps(RQ12, BC, Rrow0);
    Rrow0 = _mm_fmadd_ps(RQ22, BD, Rrow0);

    Grow0 = _mm_fmadd_ps(GQ11, AC, Grow0);
    Grow0 = _mm_fmadd_ps(GQ21, AD, Grow0);
    Grow0 = _mm_fmadd_ps(GQ12, BC, Grow0);
    Grow0 = _mm_fmadd_ps(GQ22, BD, Grow0);

    Brow0 = _mm_fmadd_ps(BQ11, AC, Brow0);
    Brow0 = _mm_fmadd_ps(BQ21, AD, Brow0);
    Brow0 = _mm_fmadd_ps(BQ12, BC, Brow0);
    Brow0 = _mm_fmadd_ps(BQ22, BD, Brow0);

    // store in to output matrix
    _mm_store_ps(intensityRout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 0 * rowSize), Rrow0);
    _mm_store_ps(intensityGout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 0 * rowSize), Grow0);
    _mm_store_ps(intensityBout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 0 * rowSize), Brow0);

    // load in coefficients for row 1
    AC = _mm_load_ps(coefficients + 16);
    AD = _mm_load_ps(coefficients + 20);
    BC = _mm_load_ps(coefficients + 24);
    BD = _mm_load_ps(coefficients + 28);

    // multiply Q11 with AC
    Rrow1 = _mm_fmadd_ps(RQ11, AC, Rrow1);
    Rrow1 = _mm_fmadd_ps(RQ21, AD, Rrow1);
    Rrow1 = _mm_fmadd_ps(RQ12, BC, Rrow1);
    Rrow1 = _mm_fmadd_ps(RQ22, BD, Rrow1);

    Grow1 = _mm_fmadd_ps(GQ11, AC, Grow1);
    Grow1 = _mm_fmadd_ps(GQ21, AD, Grow1);
    Grow1 = _mm_fmadd_ps(GQ12, BC, Grow1);
    Grow1 = _mm_fmadd_ps(GQ22, BD, Grow1);

    Brow1 = _mm_fmadd_ps(BQ11, AC, Brow1);
    Brow1 = _mm_fmadd_ps(BQ21, AD, Brow1);
    Brow1 = _mm_fmadd_ps(BQ12, BC, Brow1);
    Brow1 = _mm_fmadd_ps(BQ22, BD, Brow1);

    // store in to output matrix
    _mm_store_ps(intensityRout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 1 * rowSize), Rrow1);
    _mm_store_ps(intensityGout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 1 * rowSize), Grow1);
    _mm_store_ps(intensityBout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 1 * rowSize), Brow1);

    // load in coefficients for row 2
    AC = _mm_load_ps(coefficients + 32);
    AD = _mm_load_ps(coefficients + 36);
    BC = _mm_load_ps(coefficients + 40);
    BD = _mm_load_ps(coefficients + 44);

    // multiply Q11 with AC
    Rrow2 = _mm_fmadd_ps(RQ11, AC, Rrow2);
    Rrow2 = _mm_fmadd_ps(RQ21, AD, Rrow2);
    Rrow2 = _mm_fmadd_ps(RQ12, BC, Rrow2);
    Rrow2 = _mm_fmadd_ps(RQ22, BD, Rrow2);

    Grow2 = _mm_fmadd_ps(GQ11, AC, Grow2);
    Grow2 = _mm_fmadd_ps(GQ21, AD, Grow2);
    Grow2 = _mm_fmadd_ps(GQ12, BC, Grow2);
    Grow2 = _mm_fmadd_ps(GQ22, BD, Grow2);

    Brow2 = _mm_fmadd_ps(BQ11, AC, Brow2);
    Brow2 = _mm_fmadd_ps(BQ21, AD, Brow2);
    Brow2 = _mm_fmadd_ps(BQ12, BC, Brow2);
    Brow2 = _mm_fmadd_ps(BQ22, BD, Brow2);

    // store in to output matrix
    _mm_store_ps(intensityRout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 2 * rowSize), Rrow2);
    _mm_store_ps(intensityGout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 2 * rowSize), Grow2);
    _mm_store_ps(intensityBout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 2 * rowSize), Brow2);

    // load in coefficients for row 3
    AC = _mm_load_ps(coefficients + 48);
    AD = _mm_load_ps(coefficients + 52);
    BC = _mm_load_ps(coefficients + 56);
    BD = _mm_load_ps(coefficients + 60);

    // multiply Q11 with AC
    Rrow3 = _mm_fmadd_ps(RQ11, AC, Rrow3);
    Rrow3 = _mm_fmadd_ps(RQ21, AD, Rrow3);
    Rrow3 = _mm_fmadd_ps(RQ12, BC, Rrow3);
    Rrow3 = _mm_fmadd_ps(RQ22, BD, Rrow3);

    Grow3 = _mm_fmadd_ps(GQ11, AC, Grow3);
    Grow3 = _mm_fmadd_ps(GQ21, AD, Grow3);
    Grow3 = _mm_fmadd_ps(GQ12, BC, Grow3);
    Grow3 = _mm_fmadd_ps(GQ22, BD, Grow3);

    Brow3 = _mm_fmadd_ps(BQ11, AC, Brow3);
    Brow3 = _mm_fmadd_ps(BQ21, AD, Brow3);
    Brow3 = _mm_fmadd_ps(BQ12, BC, Brow3);
    Brow3 = _mm_fmadd_ps(BQ22, BD, Brow3);

    // store in to output matrix
    _mm_store_ps(intensityRout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 3 * rowSize), Rrow3);
    _mm_store_ps(intensityGout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 3 * rowSize), Grow3);
    _mm_store_ps(intensityBout + (((rowSize / 2) * (colSize / 2) * 2 * row_i) + ((rowSize / 2) * col_i) + 3 * rowSize), Brow3);
}

int main(int argc, char **argv)
{
    unsigned long long t0, t1;
    // kernel width
    // int outputRowSize = 1280;
    // int outputColumnSize = 960;
    int outputRowSize = 8;
    int outputColumnSize = 8;
    int inputIndex, outputRow, outputColumn, outputIndex;

    // Output Image Stack defintion
    float *outputR = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    float *outputG = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));
    float *outputB = (float *)calloc(outputRowSize * outputColumnSize, sizeof(float));

    // read in input 2x2 pixels
    // float *inputImageR = (float *)calloc(640 * 480, sizeof(float));
    // float *inputImageG = (float *)calloc(640 * 480, sizeof(float));
    // float *inputImageB = (float *)calloc(640 * 480, sizeof(float));
    float *inputImageR = (float *)calloc(ROWS * COLS, sizeof(float));
    float *inputImageG = (float *)calloc(ROWS * COLS, sizeof(float));
    float *inputImageB = (float *)calloc(ROWS * COLS, sizeof(float));

    char AfileName[100];
    strcpy(AfileName, home);
    // strcat(AfileName, "inputs/640x480.jpg");
    strcat(AfileName, "inputs/8x8.jpg");
    cout << "before decode" << endl;
    decodeImage(inputImageR, inputImageG, inputImageB, AfileName);

    for (int i = 0; i < ROWS * COLS; i++)
    {
        if (i % 4 == 0)
        {
            cout << endl;
        }
        cout << inputImageR[i] << "\t";
    }

    cout << endl;
    float *coefficients = (float *)calloc(4 * 4 * 4, sizeof(float));
    generateCoefficients(coefficients);

    t0 = rdtsc();
    // generate coefficients

    for (int col_i = 0; col_i < COLS / 2; col_i++)
    {
        for (int row_i = 0; row_i < ROWS / 2; row_i++)
        {
            kernel(inputImageR, inputImageG, inputImageB, outputR, outputG, outputB, row_i, col_i, coefficients, 2 * ROWS, 2 * COLS);
        }
    }

    for (int i = 0; i < ROWS * COLS * 4; i++)
    {
        if (i % (2 * COLS) == 0)
        {
            cout << endl;
        }
        cout << outputR[i] << "\t";
    }
    // kernel(inputImageR, inputImageG, inputImageB, outputR, outputG, outputB, coefficients, rowSize);
    t1 = rdtsc();
    printf("cycles taken (I think): %f\n", ((double)(t1 - t0) * MAX_FREQ / BASE_FREQ));

    char BfileName[100];
    strcpy(BfileName, home);
    strcat(BfileName, "results/bl/my/8x8-16x16.jpg");

    encodeImage(outputR, outputG, outputB, BfileName);
    // cout << "R=========================\n";
    // for (int i = 0; i < outputColumnSize*outputRowSize;)
    // {
    //     for (int j = 0; j < outputRowSize; j++)
    //     {
    //         cout << outputR[i++] << '\t';
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
    free(coefficients);
    free(outputR);
    free(outputG);
    free(outputB);
}
