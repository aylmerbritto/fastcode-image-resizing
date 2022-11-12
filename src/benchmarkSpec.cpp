#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

static __inline__ unsigned long long rdtsc(void)
{
	unsigned hi, lo;
	__asm__ __volatile__("rdtsc"
						 : "=a"(lo), "=d"(hi));
	return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

#define NUM_OF_IMAGES 1
#define NUM_OF_IMAGE_CONVERSIONS 1

int main(int argc, char **argv)
{
	Mat images[1];
	Mat image, result;
	char *fileName = "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/640x480.jpg";
	char *var = "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/results/benchmark/640x480-nn.jpg";
	unsigned long long startTime, endTime;
	image = imread(fileName);
	startTime = rdtsc();
	resize(image, result, Size(1280, 960), INTER_LINEAR);
	// Resizing the image - benchmark part
	endTime = rdtsc() - startTime;
	imwrite(var, result);
	cout << endTime << "\n";
	return 0;
}
