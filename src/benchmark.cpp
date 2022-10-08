#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;
// 640x480, 1920x1440, 2400x1800, 4008x3006

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#define NUM_OF_IMAGES 4

int main( int argc, char** argv ){
	Mat images[4];
	Mat image, result;
	float count = 0,percentage;
	int compareCount = 0;
	char *fileNames[NUM_OF_IMAGES] = {
		"/afs/andrew.cmu.edu/usr18/arexhari/public/645-project/inputs/640x480.jpg",
		"/afs/andrew.cmu.edu/usr18/arexhari/public/645-project/inputs/1920x1440.jpg",
		"/afs/andrew.cmu.edu/usr18/arexhari/public/645-project/inputs/2400x1800.jpg",
		"/afs/andrew.cmu.edu/usr18/arexhari/public/645-project/inputs/4008x3006.jpg"
	};
	for(int i = 0; i < NUM_OF_IMAGES; i++){
		images[i] = imread(fileNames[i]);
	}
	// int fileSizes[NUM_OF_IMAGES][2] = {
	// 	{640, 480},
	// 	{1920, 1440},
	// 	{2400, 1800},
	// 	{4008, 3006},
	// };
	int fileSizes[NUM_OF_IMAGES][2] = {
		{ 480, 640},
		{ 1440, 1920},
		{ 1800, 2400},
		{ 3006, 4008},
	};
	for(int i = 0; i < NUM_OF_IMAGES; i++){
		compareCount = 0;
		for(int j = 0; j < NUM_OF_IMAGES; j++){
			image = images[i];
			int down_width = fileSizes[j][1];
			int down_height = fileSizes[j][0];
			Mat resized_down;
			unsigned long long startTime, endTime;
			startTime = rdtsc();
			resize(image, resized_down, Size(down_width, down_height), INTER_LANCZOS4);
			endTime = rdtsc()-startTime;
			std::ostringstream oss;
			oss << "/afs/andrew.cmu.edu/usr18/arexhari/public/645-project/results/benchmark/" <<image.rows<<"x"<< image.cols <<"-"<<fileSizes[j][0] << "x" << fileSizes[j][1]<<".jpg";
			std::string var = oss.str();
			cv::compare(resized_down , images[compareCount++]  , result , cv::CMP_NE );
			count = 0.0;
			for (int p = 0; p < result.cols; p++ ) {
				for (int q = 0; q < result.rows; q++) {
					if (result.at<uchar>(q, p) == 0) {  
						count++;     // Do your operations
					}
				}
			}
			percentage = count/(result.cols*result.rows);
			cout << image.rows<<","<<fileSizes[j][0] <<","<< endTime<<","<< percentage <<"\n";
			imwrite(var, resized_down);
		}
	} 
	return 0;
}

