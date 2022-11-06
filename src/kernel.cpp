#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <x86intrin.h>
#include "xmmintrin.h"
#include "immintrin.h"

using namespace cv;
using namespace std;

void kernelSauce(float *kernelPoints){
    /*
        TODO: To calculate Z,A,B,C,D
        The Values are going to be the same throughout given the kernel input and output are the same 
    */
    float Z = 1/9;
    int kernelSauceIndex = 0, i;
    float a1[4] = {3,3,3,3}; float a2[4] = {2,2,2,2}; float a3[4] = {1,1,1,1}; float a4[4] = {0,0,0,0};
    float b4[4] = {3,3,3,3}; float b3[4] = {2,2,2,2}; float b2[4] = {1,1,1,1}; float b1[4] = {0,0,0,0};
    float c[4] = {3, 2, 1, 0}; float d[4] = {0, 1, 2, 3};
    __m128 vec1, vec2, vec3, vec4;
    // float *result = (float*)calloc(4,sizeof(float));
    /*
        a1*c, a2*c, a3*c, a4*c 
        a1*d, a2*d, a3*d, a4*d 
        b1*c, b2*c, b3*c, b4*c 
        b1*d, b2*d, b3*d, b4*d
    */
    vec1 = _mm_set_ps(0, 1/3.0, 2/3.0, 1); //z*a1*c
    vec2 = _mm_set_ps(0, 2/9.0, 4/9.0, 6/9.0); //z*a2*c
    vec3 = _mm_set_ps(0, 1/9.0, 2/9.0, 3/9.0); //z*a3*c
    vec4 = _mm_set_ps(0, 0, 0, 0); //z*a4*c
    _mm_store_ps(kernelPoints, vec1);
    _mm_store_ps(kernelPoints + 4, vec2);
    _mm_store_ps(kernelPoints + 8, vec3);
    _mm_store_ps(kernelPoints + 12, vec4);
    

    vec1 = _mm_set_ps(1, 2/3.0, 1/3.0, 0); //z*a1*d
    vec2 = _mm_set_ps(6/9.0, 4/9.0, 2/9.0, 0); //z*a2*d
    vec3 = _mm_set_ps(3/9.0, 2/9.0, 1/9.0, 0); //z*a3*d
    vec4 = _mm_set_ps(0, 0, 0, 0); //z*a4*c
    _mm_store_ps(kernelPoints + 16, vec1);
    _mm_store_ps(kernelPoints + 20, vec2);
    _mm_store_ps(kernelPoints + 24, vec3);
    _mm_store_ps(kernelPoints + 28, vec4);

    vec4 = _mm_set_ps(0, 1/3.0, 2/3.0, 1); //z*b4*c
    vec3 = _mm_set_ps(0, 2/9.0, 4/9.0, 6/9.0); //z*b3*c
    vec2 = _mm_set_ps(0, 1/9.0, 2/9.0, 3/9.0); //z*b2*c
    vec1 = _mm_set_ps(0, 0, 0, 0); //z*b1*c
    _mm_store_ps(kernelPoints + 32, vec1);
    _mm_store_ps(kernelPoints + 36, vec2);
    _mm_store_ps(kernelPoints + 40, vec3);
    _mm_store_ps(kernelPoints + 44, vec4);
    
    vec4 = _mm_set_ps(1.0, 2/3.0, 1/3.0, 0.0); //z*b4*d
    vec3 = _mm_set_ps(6/9.0, 4/9.0, 2/9.0, 0.0); //z*b3*d
    vec2 = _mm_set_ps(3/9.0, 2/9.0, 1/9.0, 0.0); //z*b2*d
    vec1 = _mm_set_ps(0.0, 0.0, 0.0, 0.0); //z*b1*d
    _mm_store_ps(kernelPoints + 48, vec1);
    _mm_store_ps(kernelPoints + 52, vec2);
    _mm_store_ps(kernelPoints + 56, vec3);
    _mm_store_ps(kernelPoints + 60, vec4);
}

void kernel(float *Ri, float *Gi, float *Bi, float *Ro, float *Go, float *Bo, float *kernelPoints, int rowSize){
    int i;
    __m128 Rin, R, Gin, G, Bin, B, K1, K2, rOut1, rOut2, gOut1, gOut2, bOut1, bOut2;
    Rin = _mm_load_ps(Ri); Gin = _mm_load_ps(Gi); Bin = _mm_load_ps(Bi);
    K1 = _mm_load_ps(kernelPoints); K2 = _mm_load_ps(kernelPoints+4);
    rOut1 = _mm_load_ps(Ro); rOut2 = _mm_load_ps(Ro+4);
    gOut1 = _mm_load_ps(Go); gOut2 = _mm_load_ps(Go+4);
    bOut1 = _mm_load_ps(Bo); bOut2 = _mm_load_ps(Bo+4);
     
    const int mask1 = (0) | (0 << 2) | (0 << 4) | (0 << 6);
    R = _mm_permute_ps(Rin, mask1);
    G = _mm_permute_ps(Gin, mask1);
    B = _mm_permute_ps(Bin, mask1);
    rOut1 = _mm_fmadd_ps(R, K1, rOut1); rOut2 = _mm_fmadd_ps(R, K2, rOut2);
    gOut1 = _mm_fmadd_ps(G, K1, gOut1); gOut2 = _mm_fmadd_ps(G, K2, gOut2);
    bOut1 = _mm_fmadd_ps(B, K1, bOut1); bOut2 = _mm_fmadd_ps(B, K2, bOut2);
    
    
    const int mask2 = (1) | (1 << 2) | (1 << 4) | (1 << 6);
    R = _mm_permute_ps(Rin, mask2);
    G = _mm_permute_ps(Gin, mask2);
    B = _mm_permute_ps(Bin, mask2);
    K1 = _mm_load_ps(kernelPoints+16); K2 = _mm_load_ps(kernelPoints+20);
    rOut1 = _mm_fmadd_ps(R, K1, rOut1); rOut2 = _mm_fmadd_ps(R, K2, rOut2);
    gOut1 = _mm_fmadd_ps(G, K1, gOut1); gOut2 = _mm_fmadd_ps(G, K2, gOut2);
    bOut1 = _mm_fmadd_ps(B, K1, bOut1); bOut2 = _mm_fmadd_ps(B, K2, bOut2);
  

    const int mask3 = (2) | (2 << 2) | (2 << 4) | (2 << 6);
    R = _mm_permute_ps(Rin, mask3);
    G = _mm_permute_ps(Gin, mask3);
    B = _mm_permute_ps(Bin, mask3);
    K1 = _mm_load_ps(kernelPoints+32); K2 = _mm_load_ps(kernelPoints+36);
    rOut1 = _mm_fmadd_ps(R, K1, rOut1); rOut2 = _mm_fmadd_ps(R, K2, rOut2);
    gOut1 = _mm_fmadd_ps(G, K1, gOut1); gOut2 = _mm_fmadd_ps(G, K2, gOut2);
    bOut1 = _mm_fmadd_ps(B, K1, bOut1); bOut2 = _mm_fmadd_ps(B, K2, bOut2);


    const int mask4 = (3) | (3 << 2) | (3 << 4) | (3 << 6);
    R = _mm_permute_ps(Rin, mask4);
    G = _mm_permute_ps(Gin, mask4);
    B = _mm_permute_ps(Bin, mask4);
    K1 = _mm_load_ps(kernelPoints+48); K2 = _mm_load_ps(kernelPoints+52);
    rOut1 = _mm_fmadd_ps(R, K1, rOut1); rOut2 = _mm_fmadd_ps(R, K2, rOut2);    
    gOut1 = _mm_fmadd_ps(G, K1, gOut1); gOut2 = _mm_fmadd_ps(G, K2, gOut2);    
    bOut1 = _mm_fmadd_ps(B, K1, bOut1); bOut2 = _mm_fmadd_ps(B, K2, bOut2);


    _mm_store_ps(Ro, rOut1); _mm_store_ps(Ro+rowSize , rOut2);
    _mm_store_ps(Go, gOut1); _mm_store_ps(Go+rowSize , gOut2);
    _mm_store_ps(Bo, bOut1); _mm_store_ps(Bo+rowSize , bOut2);

    //==========================================================================

    K1 = _mm_load_ps(kernelPoints+8); K2 = _mm_load_ps(kernelPoints+12);
    rOut1 = _mm_load_ps(Ro+8); rOut2 = _mm_load_ps(Ro+12);
    gOut1 = _mm_load_ps(Go+8); gOut2 = _mm_load_ps(Go+12);
    bOut1 = _mm_load_ps(Bo+8); bOut2 = _mm_load_ps(Bo+12);
    
    R = _mm_permute_ps(Rin, mask1);
    G = _mm_permute_ps(Gin, mask1);
    B = _mm_permute_ps(Bin, mask1);
    rOut1 = _mm_fmadd_ps(R, K1, rOut1); rOut2 = _mm_fmadd_ps(R, K2, rOut2);
    gOut1 = _mm_fmadd_ps(G, K1, gOut1); gOut2 = _mm_fmadd_ps(G, K2, gOut2);
    bOut1 = _mm_fmadd_ps(B, K1, bOut1); bOut2 = _mm_fmadd_ps(B, K2, bOut2);


    R = _mm_permute_ps(Rin, mask2);
    G = _mm_permute_ps(Gin, mask2);
    B = _mm_permute_ps(Bin, mask2);
    K1 = _mm_load_ps(kernelPoints+24); K2 = _mm_load_ps(kernelPoints+28);
    rOut1 = _mm_fmadd_ps(R, K1, rOut1); rOut2 = _mm_fmadd_ps(R, K2, rOut2);
    gOut1 = _mm_fmadd_ps(G, K1, gOut1); gOut2 = _mm_fmadd_ps(G, K2, gOut2);
    bOut1 = _mm_fmadd_ps(B, K1, bOut1); bOut2 = _mm_fmadd_ps(B, K2, bOut2);

    
    R = _mm_permute_ps(Rin, mask3);
    G = _mm_permute_ps(Gin, mask3);
    B = _mm_permute_ps(Bin, mask3);
    K1 = _mm_load_ps(kernelPoints+40); K2 = _mm_load_ps(kernelPoints+44);
    rOut1 = _mm_fmadd_ps(R, K1, rOut1); rOut2 = _mm_fmadd_ps(R, K2, rOut2);
    gOut1 = _mm_fmadd_ps(G, K1, gOut1); gOut2 = _mm_fmadd_ps(G, K2, gOut2);
    bOut1 = _mm_fmadd_ps(B, K1, bOut1); bOut2 = _mm_fmadd_ps(B, K2, bOut2);


    R = _mm_permute_ps(Rin, mask4);
    G = _mm_permute_ps(Gin, mask4);
    B = _mm_permute_ps(Bin, mask4);
    K1 = _mm_load_ps(kernelPoints+56); K2 = _mm_load_ps(kernelPoints+60);
    rOut1 = _mm_fmadd_ps(R, K1, rOut1); rOut2 = _mm_fmadd_ps(R, K2, rOut2);    
    gOut1 = _mm_fmadd_ps(G, K1, gOut1); gOut2 = _mm_fmadd_ps(G, K2, gOut2);    
    bOut1 = _mm_fmadd_ps(B, K1, bOut1); bOut2 = _mm_fmadd_ps(B, K2, bOut2);
    
    _mm_store_ps(Ro+(2*rowSize), rOut1); _mm_store_ps(Ro+(3*rowSize) , rOut2);
    _mm_store_ps(Go+(2*rowSize), gOut1); _mm_store_ps(Go+(3*rowSize) , gOut2);
    _mm_store_ps(Bo+(2*rowSize), bOut1); _mm_store_ps(Bo+(3*rowSize) , bOut2);
}

int main( int argc, char** argv ){
    //Image Information stack
    int rowSize = 4;

    //Algorithm Logistics Stack defintion
    int i, j;
    float inputImageR[4] = {1,2,3,4};
    float inputImageG[4] = {5,8,8,11};
    float inputImageB[4] = {6,9,9,12};
    float *outputImage = (float*)calloc(48,sizeof(float));
    float *kernelPoints = (float*)malloc(64*sizeof(float));
    
    //Output Image Stack defintion
    float *outputR = (float*)calloc(16,sizeof(float));
    float *outputG = (float*)calloc(16,sizeof(float));
    float *outputB = (float*)calloc(16,sizeof(float));
    kernelSauce(kernelPoints);


    kernel(inputImageR,inputImageG,inputImageB,outputR, outputG, outputB, kernelPoints, rowSize);
    cout << "R=========================\n";
    for(i=0;i<16;){
        for(j=0;j<4;j++){
            cout<<outputR[i++]<<'\t';
        }
        cout<<'\n';
    }
    cout << "G=========================\n";
    for(i=0;i<16;i){
        for(j=0;j<4;j++){
            cout<<outputG[i++]<<'\t';
        }
        cout<<'\n';
    }
    cout << "B=========================\n";
    for(i=0;i<16;i){
        for(j=0;j<4;j++){
            cout<<outputB[i++]<<'\t';
        }
        cout<<'\n';
    }
    free(outputImage);
    free(kernelPoints);
    free(outputR);
    free(outputG);
    free(outputB);
}
