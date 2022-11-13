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
#define WINDOWSIZE 2
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

int main(int argc, char **argv)
{
	// Mat images[4];
	Mat images[1];
	Mat image, result;
	float count = 0, percentage;
	int compareCount = 0;
	// char *fileNames[NUM_OF_IMAGES] = {
	// 	"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/640x480.jpg",
	// 	"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/1920x1440.jpg",
	// 	"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/2400x1800.jpg",
	// 	"/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/4008x3006.jpg"};
	char *fileNames[NUM_OF_IMAGES] = {
		"/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/inputs/8x8.jpg"};
	for (int i = 0; i < NUM_OF_IMAGES; i++)
	{
		images[i] = imread(fileNames[i]);
	}
	// int fileSizes[NUM_OF_IMAGE_CONVERSIONS][2] = {
	// 	{480, 640},
	// 	{960, 1280},
	// 	{1440, 1920},
	// 	{1800, 2400},
	// 	{3006, 4008},
	// 	{6012, 8004},
	// 	{12042, 16008},
	// };
	int fileSizes[NUM_OF_IMAGE_CONVERSIONS][2] = {
		{16, 16},
	};
	// for (int i = 0; i < NUM_OF_IMAGES; i++)
	// {
	// 	compareCount = 0;
	// 	for (int j = i + 1; j < NUM_OF_IMAGE_CONVERSIONS; j++)
	// 	{
	// image = images[i];
	image = imread("/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/inputs/8x8.jpg");
	// int down_width = fileSizes[j][1];
	// int down_height = fileSizes[j][0];
	int down_width = 16;
	int down_height = 16;
	Mat resized_down;
	unsigned long long startTime;
	unsigned long long endTime;
	startTime = rdtsc();
	// resize(image, resized_down, Size(down_width, down_height), INTER_LINEAR);
	resize(image, resized_down, Size(down_width, down_height), INTER_LINEAR);
	// Resizing the image - benchmark part
	endTime = rdtsc() - startTime;
	std::ostringstream oss;

	// oss << home << "results/bl/cv/" << image.cols << "x" << image.rows << "-" << fileSizes[j][1] << "x" << fileSizes[j][0] << ".jpg";
	// std::string var = oss.str();
	// cout << image.rows << "," << fileSizes[j][0] << "," << endTime << "\n";
	// imwrite(var, resized_down);
	imwrite("/afs/andrew.cmu.edu/usr19/anesathu/private/fastCodeProject/results/bl/cv/8x8-16x16.jpg", resized_down);
	// 	}
	// }

	// validate that the output for openCV and our algorithm are the same
	// read in input 2x2 pixels
	// float *AinputImageR = (float *)calloc(1280 * 960, sizeof(float));
	// float *AinputImageG = (float *)calloc(1280 * 960, sizeof(float));
	// float *AinputImageB = (float *)calloc(1280 * 960, sizeof(float));
	float *AinputImageR = (float *)calloc(16 * 16, sizeof(float));
	float *AinputImageG = (float *)calloc(16 * 16, sizeof(float));
	float *AinputImageB = (float *)calloc(16 * 16, sizeof(float));
	char AfileName[100];
	strcpy(AfileName, home);
	// strcat(AfileName, "results/bl/my/640x480-1280x960.jpg");
	strcat(AfileName, "results/bl/my/8x8-16x16.jpg");

	cout << AfileName << endl;
	decodeImage(AinputImageR, AinputImageG, AinputImageB, AfileName);

	// float *BinputImageR = (float *)calloc(1280 * 960, sizeof(float));
	// float *BinputImageG = (float *)calloc(1280 * 960, sizeof(float));
	// float *BinputImageB = (float *)calloc(1280 * 960, sizeof(float));
	float *BinputImageR = (float *)calloc(16 * 16, sizeof(float));
	float *BinputImageG = (float *)calloc(16 * 16, sizeof(float));
	float *BinputImageB = (float *)calloc(16 * 16, sizeof(float));
	char BfileName[100];
	strcpy(BfileName, home);
	// strcat(BfileName, "results/bl/cv/640x480-1280x960.jpg");
	strcat(BfileName, "results/bl/cv/8x8-16x16.jpg");

	cout << BfileName << endl;
	decodeImage(BinputImageR, BinputImageG, BinputImageB, BfileName);

	bool correct = true;
	for (int i = 0; i < 16 * 16; i++)
	{
		if (i % 16 == 0)
		{
			cout << endl;
		}
		// if (abs(AinputImageR[i] - BinputImageR[i]) > 0.1)
		// {
		// correct = false;
		cout << abs(AinputImageR[i] - BinputImageR[i]) << "\t";
		// cout << "image A and B are not the same" << endl;
		// break;
		// }
		// if (abs(AinputImageG[i] - BinputImageG[i]) > 0.1)
		// {
		// correct = false;
		// cout << abs(AinputImageG[i] - BinputImageG[i]) << "\t";
		// cout <<  abs(AinputImageR[i] - BinputImageR[i]) << "\t";
		// cout << "image A and B are not the same" << endl;
		// break;
		// }
		// if (abs(AinputImageB[i] - BinputImageB[i]) > 0.1)
		// {
		// 	correct = false;
		// cout << abs(AinputImageB[i] - BinputImageB[i]) << "\t";
		// cout << "image A and B are not the same" << endl;
		// break;
		// }
	}
	cout << endl;
	// if (correct)
	// {
	// 	cout << "images A and B are the same" << endl;
	// }
	// else
	// {
	// 	cout << "images A and B are not the same" << endl;
	// }

	return 0;
}
