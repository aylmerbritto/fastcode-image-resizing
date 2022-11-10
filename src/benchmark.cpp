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

#define NUM_OF_IMAGES 4
#define NUM_OF_IMAGE_CONVERSIONS 6

int main(int argc, char **argv)
{
	Mat images[4];
	Mat image, result;
	float count = 0, percentage;
	int compareCount = 0;
	char *fileNames[NUM_OF_IMAGES] = {
		"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/640x480.jpg",
		"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/1920x1440.jpg",
		"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/2400x1800.jpg",
		"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/4008x3006.jpg"};
	for (int i = 0; i < NUM_OF_IMAGES; i++)
	{
		images[i] = imread(fileNames[i]);
	}
	int fileSizes[NUM_OF_IMAGE_CONVERSIONS][2] = {
		{480, 640},
		{1440, 1920},
		{1800, 2400},
		{3006, 4008},
		{6012, 8004},
		{12042, 16008},
	};
	for (int i = 0; i < NUM_OF_IMAGES; i++)
	{
		compareCount = 0;
		for (int j = i + 1; j < NUM_OF_IMAGE_CONVERSIONS; j++)
		{
			image = images[i];
			int down_width = fileSizes[j][1];
			int down_height = fileSizes[j][0];
			Mat resized_down;
			unsigned long long startTime, endTime;
			startTime = rdtsc();
			resize(image, resized_down, Size(down_width, down_height), INTER_NEAREST);
			// Resizing the image - benchmark part
			endTime = rdtsc() - startTime;
			std::ostringstream oss;
			oss << "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/results/benchmark/" << image.rows << "x" << image.cols << "-" << fileSizes[j][0] << "x" << fileSizes[j][1] << ".jpg";
			std::string var = oss.str();
			cout << image.rows << "," << fileSizes[j][0] << "," << endTime << "\n";
			imwrite(var, resized_down);
		}
	}
	return 0;
}
