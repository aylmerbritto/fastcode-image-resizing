#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include "omp.h"

using namespace cv;
using namespace std;

#define MAX_FREQ 3.2
#define BASE_FREQ 2.4

static __inline__ unsigned long long rdtsc(void)
{
	unsigned hi, lo;
	__asm__ __volatile__("rdtsc"
						 : "=a"(lo), "=d"(hi));
	return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int main(int argc, char **argv)
{
	Mat images[1];
	Mat image, result;
	char *fileName = argv[1];
	unsigned long long startTime, endTime;
	image = imread(fileName);
	startTime = rdtsc();
	resize(image, result, Size(image.cols*2, image.rows*2), INTER_LINEAR);
	endTime = (rdtsc() - startTime)* MAX_FREQ / BASE_FREQ;
	cout <<j<<','<< endTime << endl;
	return 0;
}
