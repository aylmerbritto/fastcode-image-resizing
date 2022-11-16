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

#define NUM_OF_IMAGES 2
#define NUM_OF_IMAGE_CONVERSIONS 1
#define WINDOWSIZE 2
// mode is either INTER_LINEAR or INTER_NEAREST
#define MODE INTER_LINEAR
char *home = "/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/";
// 	"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/640x480.jpg",
//
// 	"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/2400x1800.jpg",
// 	"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/4008x3006.jpg"};
char *fileNames[NUM_OF_IMAGES] = {
    "/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/inputs/640x480.jpg",
    "/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/inputs/1920x1440.jpg"};

int upscalingMultipliers[NUM_OF_IMAGE_CONVERSIONS] = {2};

// timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc"
                         : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int decodeImage(float *inputImageR, float *inputImageG, float *inputImageB, const char *fileName)
{
    int i, j, index = 0;
    float *tmpBuffer;
    // READ IMAGE and Init buffers
    Mat fullImage;
    Mat windowImage;
    Mat channels[3];
    std::vector<float> array;
    fullImage = imread(fileName);
    int imageRows = (int)fullImage.rows;
    int imageCols = (int)fullImage.cols;
    cout << "Width : " << imageCols << endl;
    cout << "Height: " << imageRows << endl;

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

int encodeImage(float *outputR, float *outputG, float *outputB, const char *fileName, int outWidth, int outHeight)
{
    vector<Mat> channels;
    Mat finalImage;
    Mat out;
    cv::Mat matR = cv::Mat(outHeight, outWidth, CV_32F, outputR);
    cv::Mat matG = cv::Mat(outHeight, outWidth, CV_32F, outputG);
    cv::Mat matB = cv::Mat(outHeight, outWidth, CV_32F, outputB);

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

    float vec1flt[4] = {9.0 / 9.0, 6.0 / 9.0, 3.0 / 9.0, 0.0 / 9.0};
    float vec2flt[4] = {0.0 / 9.0, 3.0 / 9.0, 6.0 / 3.0, 9.0 / 9.0};
    float vec3flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    float vec4flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};

    vec1 = _mm_load_ps(vec1flt);
    vec2 = _mm_load_ps(vec2flt);
    vec3 = _mm_load_ps(vec3flt);
    vec4 = _mm_load_ps(vec4flt);

    _mm_store_ps(coefficients, vec1);
    _mm_store_ps(coefficients + 4, vec2);
    _mm_store_ps(coefficients + 8, vec3);
    _mm_store_ps(coefficients + 12, vec4);

    float vec5flt[4] = {6.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0, 0.0 / 9.0};
    float vec6flt[4] = {0.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0, 6.0 / 9.0};
    float vec7flt[4] = {3.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0, 0.0 / 9.0};
    float vec8flt[4] = {0.0 / 9.0, 1.0 / 9.0, 2.0 / 9.0, 3.0 / 9.0};

    vec1 = _mm_load_ps(vec5flt);
    vec2 = _mm_load_ps(vec6flt);
    vec3 = _mm_load_ps(vec7flt);
    vec4 = _mm_load_ps(vec8flt);

    _mm_store_ps(coefficients + 16, vec1);
    _mm_store_ps(coefficients + 20, vec2);
    _mm_store_ps(coefficients + 24, vec3);
    _mm_store_ps(coefficients + 28, vec4);

    float vec9flt[4] = {3.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0, 0.0 / 9.0};
    float vec10flt[4] = {0.0 / 9.0, 1.0 / 9.0, 2.0 / 9.0, 3.0 / 9.0};
    float vec11flt[4] = {6.0 / 9.0, 4.0 / 9.0, 2.0 / 4.0, 0.0 / 9.0};
    float vec12flt[4] = {0.0 / 9.0, 2.0 / 4.0, 4.0 / 9.0, 6.0 / 9.0};

    vec1 = _mm_load_ps(vec9flt);
    vec2 = _mm_load_ps(vec10flt);
    vec3 = _mm_load_ps(vec11flt);
    vec4 = _mm_load_ps(vec12flt);

    _mm_store_ps(coefficients + 32, vec1);
    _mm_store_ps(coefficients + 36, vec2);
    _mm_store_ps(coefficients + 40, vec3);
    _mm_store_ps(coefficients + 44, vec4);

    float vec13flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    float vec14flt[4] = {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0};
    float vec15flt[4] = {9.0 / 9.0, 6.0 / 9.0, 3.0 / 9.0, 0.0 / 9.0};
    float vec16flt[4] = {0.0 / 9.0, 3.0 / 9.0, 6.0 / 9.0, 9.0 / 9.0};

    vec1 = _mm_load_ps(vec13flt);
    vec2 = _mm_load_ps(vec14flt);
    vec3 = _mm_load_ps(vec15flt);
    vec4 = _mm_load_ps(vec16flt);

    _mm_store_ps(coefficients + 48, vec1);
    _mm_store_ps(coefficients + 52, vec2);
    _mm_store_ps(coefficients + 56, vec3);
    _mm_store_ps(coefficients + 60, vec4);
}

void kernel(const float *intensityRin, const float *intensityGin, const float *intensityBin, float *intensityRout, float *intensityGout, float *intensityBout, int row_i, int col_i, const float *coefficients, int rowSize, int colSize)
{

    __m128 RQ11, RQ21, RQ12, RQ22;
    __m128 GQ11, GQ21, GQ12, GQ22;
    __m128 BQ11, BQ21, BQ12, BQ22;

    __m128 Rrow0, Rrow1, Rrow2, Rrow3;
    __m128 Grow0, Grow1, Grow2, Grow3;
    __m128 Brow0, Brow1, Brow2, Brow3;

    __m128 AC, AD, BC, BD;

    RQ11 = _mm_broadcast_ss(&intensityRin[(colSize)*row_i + (4) * col_i + 0]);
    RQ21 = _mm_broadcast_ss(&intensityRin[(colSize)*row_i + (4) * col_i + 1]);
    RQ12 = _mm_broadcast_ss(&intensityRin[(colSize)*row_i + (4) * col_i + 2]);
    RQ22 = _mm_broadcast_ss(&intensityRin[(colSize)*row_i + (4) * col_i + 3]);

    GQ11 = _mm_broadcast_ss(&intensityGin[(colSize)*row_i + (4) * col_i + 0]);
    GQ21 = _mm_broadcast_ss(&intensityGin[(colSize)*row_i + (4) * col_i + 1]);
    GQ12 = _mm_broadcast_ss(&intensityGin[(colSize)*row_i + (4) * col_i + 2]);
    GQ22 = _mm_broadcast_ss(&intensityGin[(colSize)*row_i + (4) * col_i + 3]);

    BQ11 = _mm_broadcast_ss(&intensityBin[(colSize)*row_i + (4) * col_i + 0]);
    BQ21 = _mm_broadcast_ss(&intensityBin[(colSize)*row_i + (4) * col_i + 1]);
    BQ12 = _mm_broadcast_ss(&intensityBin[(colSize)*row_i + (4) * col_i + 2]);
    BQ22 = _mm_broadcast_ss(&intensityBin[(colSize)*row_i + (4) * col_i + 3]);

    Rrow0 = _mm_load_ps(intensityRout + (((4) * (colSize)*row_i) + ((4) * col_i) + 0 * colSize));
    Grow0 = _mm_load_ps(intensityGout + (((4) * (colSize)*row_i) + ((4) * col_i) + 0 * colSize));
    Brow0 = _mm_load_ps(intensityBout + (((4) * (colSize)*row_i) + ((4) * col_i) + 0 * colSize));

    Rrow1 = _mm_load_ps(intensityRout + (((4) * (colSize)*row_i) + ((4) * col_i) + 1 * colSize));
    Grow1 = _mm_load_ps(intensityGout + (((4) * (colSize)*row_i) + ((4) * col_i) + 1 * colSize));
    Brow1 = _mm_load_ps(intensityBout + (((4) * (colSize)*row_i) + ((4) * col_i) + 1 * colSize));

    Rrow2 = _mm_load_ps(intensityRout + (((4) * (colSize)*row_i) + ((4) * col_i) + 2 * colSize));
    Grow2 = _mm_load_ps(intensityGout + (((4) * (colSize)*row_i) + ((4) * col_i) + 2 * colSize));
    Brow2 = _mm_load_ps(intensityBout + (((4) * (colSize)*row_i) + ((4) * col_i) + 2 * colSize));

    Rrow3 = _mm_load_ps(intensityRout + (((4) * (colSize)*row_i) + ((4) * col_i) + 3 * colSize));
    Grow3 = _mm_load_ps(intensityGout + (((4) * (colSize)*row_i) + ((4) * col_i) + 3 * colSize));
    Brow3 = _mm_load_ps(intensityBout + (((4) * (colSize)*row_i) + ((4) * col_i) + 3 * colSize));

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
    _mm_store_ps(intensityRout + (((4) * (colSize)*row_i) + ((4) * col_i) + 0 * colSize), Rrow0);
    _mm_store_ps(intensityGout + (((4) * (colSize)*row_i) + ((4) * col_i) + 0 * colSize), Grow0);
    _mm_store_ps(intensityBout + (((4) * (colSize)*row_i) + ((4) * col_i) + 0 * colSize), Brow0);

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
    _mm_store_ps(intensityRout + (((4) * (colSize)*row_i) + ((4) * col_i) + 1 * colSize), Rrow1);
    _mm_store_ps(intensityGout + (((4) * (colSize)*row_i) + ((4) * col_i) + 1 * colSize), Grow1);
    _mm_store_ps(intensityBout + (((4) * (colSize)*row_i) + ((4) * col_i) + 1 * colSize), Brow1);

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
    _mm_store_ps(intensityRout + (((4) * (colSize)*row_i) + ((4) * col_i) + 2 * colSize), Rrow2);
    _mm_store_ps(intensityGout + (((4) * (colSize)*row_i) + ((4) * col_i) + 2 * colSize), Grow2);
    _mm_store_ps(intensityBout + (((4) * (colSize)*row_i) + ((4) * col_i) + 2 * colSize), Brow2);

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
    _mm_store_ps(intensityRout + (((4) * (colSize)*row_i) + ((4) * col_i) + 3 * colSize), Rrow3);
    _mm_store_ps(intensityGout + (((4) * (colSize)*row_i) + ((4) * col_i) + 3 * colSize), Grow3);
    _mm_store_ps(intensityBout + (((4) * (colSize)*row_i) + ((4) * col_i) + 3 * colSize), Brow3);
}

int main(int argc, char **argv)
{
    unsigned long long startTime;
    unsigned long long endTime;

    Mat images[1];
    Mat image;

    for (int i = 0; i < NUM_OF_IMAGES; i++)
    {
        images[i] = imread(fileNames[i]);
    }

    float *coefficients = (float *)calloc(4 * 4 * 4, sizeof(float));
    generateCoefficients(coefficients);

    for (int i = 0; i < NUM_OF_IMAGES; i++)
    {
        image = images[i];
        int inWidth = image.cols;
        int inHeight = image.rows;

        float *inputImageR = (float *)calloc(inHeight * inWidth, sizeof(float));
        float *inputImageG = (float *)calloc(inHeight * inWidth, sizeof(float));
        float *inputImageB = (float *)calloc(inHeight * inWidth, sizeof(float));

        cout << "input: " << fileNames[i] << endl;
        decodeImage(inputImageR, inputImageB, inputImageG, fileNames[i]);

        for (int j = 0; j < NUM_OF_IMAGE_CONVERSIONS; j++)
        {

            int outWidth = upscalingMultipliers[j] * inWidth;
            int outHeight = upscalingMultipliers[j] * inHeight;

            float *outputR = (float *)calloc(outHeight * outWidth, sizeof(float));
            float *outputG = (float *)calloc(outHeight * outWidth, sizeof(float));
            float *outputB = (float *)calloc(outHeight * outWidth, sizeof(float));

            startTime = rdtsc();
            for (int row_i = 0; row_i < outHeight / 4; row_i++)
            {
                // cout << row_i << endl;
                for (int col_i = 0; col_i < outWidth / 4; col_i++)
                {
                    kernel(inputImageR, inputImageG, inputImageB, outputR, outputG, outputB, row_i, col_i, coefficients, outHeight, outWidth);
                }
            }
            endTime = rdtsc();

            // for (int i = 0; i < outHeight * outWidth; i++)
            // {
            //     if (i % (outHeight * outWidth / 4) == 0)
            //     {
            //         cout << endl;
            //     }
            //     cout << outputR[i] << "\t";
            // }

            std::ostringstream oss;
            oss.str("");
            oss.str().clear();

            if (MODE == INTER_LINEAR)
            {
                oss << home << "results/bl/my/" << inWidth << "x" << inHeight << "-" << outWidth << "x" << outHeight << ".jpg";
            }
            else if (MODE == INTER_NEAREST)
            {
                oss << home << "results/nn/my/" << inWidth << "x" << inHeight << "-" << outWidth << "x" << outHeight << ".jpg";
            }
            else
            {
                cout << "Invalid MODE. Only supported modes are INTER_LINEAR and INTER_NEAREST" << endl;
            }

            std::string outPath = oss.str();
            cout << "output: " << outPath << endl;
            encodeImage(outputR, outputG, outputB, outPath.c_str(), outWidth, outHeight);
            oss.str("");
            oss.str().clear();
            printf("cycles taken (I think): %f\n", ((double)(endTime - startTime) * MAX_FREQ / BASE_FREQ));

            free(outputR);
            free(outputG);
            free(outputB);
        }
        free(inputImageR);
        free(inputImageG);
        free(inputImageB);
    }
    free(coefficients);
}