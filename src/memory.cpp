#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <x86intrin.h>
#include "xmmintrin.h"
#include "immintrin.h"

#define WINDOWSIZE 2

using namespace cv;
using namespace std;


 
int main(){
    int i,j,index=0;
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
    float *rChannel = (float*)calloc(fullImage.cols*fullImage.rows, sizeof(float));
    float *gChannel = (float*)calloc(fullImage.cols*fullImage.rows, sizeof(float));
    float *bChannel = (float*)calloc(fullImage.cols*fullImage.rows, sizeof(float));

    for(i=0;i+WINDOWSIZE<=imageRows;i=i+WINDOWSIZE){
        for(j=0;j+WINDOWSIZE<=imageCols;j=j+WINDOWSIZE){
            windowImage = fullImage(Range(i,i+WINDOWSIZE), Range(j,j+WINDOWSIZE));
            split(windowImage, channels);
            array.assign(channels[0].datastart, channels[0].dataend);
            tmpBuffer = &array[0];
            memcpy(bChannel+index, tmpBuffer, 4*sizeof(float));
            array.assign(channels[1].datastart, channels[1].dataend);
            tmpBuffer = &array[0];
            memcpy(gChannel+index, tmpBuffer, 4*sizeof(float));
            array.assign(channels[2].datastart, channels[2].dataend);
            tmpBuffer = &array[0];
            memcpy(rChannel+index, tmpBuffer, 4*sizeof(float));
            index = index+4;
        }
    }
    return 0;
}