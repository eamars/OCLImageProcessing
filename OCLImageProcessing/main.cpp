#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#include "utils.h"
#include "Timer.h"

#include "OCLCanny.h"
#include "CPUCanny.h"

using std::cerr;
using std::cout;
using std::string;
using std::vector;
using std::endl;
using cv::Mat;
using cv::Size;



void CannyOverallCPUTest(size_t size)
{
	
	// create random image
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(64, 25);
	unsigned char *pRandomImage = (unsigned char *)malloc(size * size);

	for (size_t pixel = 0; pixel < size * size; pixel++)
	{
		pRandomImage[pixel] = (unsigned char)std::round(d(gen));
	}

	Mat inputImage(size, size, CV_8UC1, pRandomImage);

	Timer timer;

	double totalTime = 0.0f;
	cout << "Size: " << size << " ";
	for (int tried = 0; tried < 3; tried++)
	{
		timer.start();

		CPUCanny imageProcessor;
		imageProcessor.LoadOCVImage(inputImage);

		imageProcessor.Gaussian();
		imageProcessor.Sobel();
		imageProcessor.NonMaximaSuppression();
		imageProcessor.HysteresisThresholding();
		
		timer.stop();
		totalTime += timer.getElapsedTimeInMicroSec();
	}
	cout << " Avg: " << totalTime / 3 << "us\n";
}

void CannyOverallGPUTest(size_t size)
{
	// create random image
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(64, 25);
	unsigned char *pRandomImage = (unsigned char *)malloc(size * size);

	for (size_t pixel = 0; pixel < size * size; pixel++)
	{
		pRandomImage[pixel] = (unsigned char)std::round(d(gen));
	}

	Mat inputImage(size, size, CV_8UC1, pRandomImage);

	Timer timer;

	double totalTime = 0.0f;
	cout << "Size: " << size << " ";
	for (int tried = 0; tried < 1; tried++)
	{
		timer.start();

		OCLCanny imageProcessor;
		imageProcessor.LoadOCVImage(inputImage);

		imageProcessor.Gaussian();
		imageProcessor.Sobel();
		imageProcessor.NonMaximaSuppression();
		imageProcessor.HysteresisThresholding();
		Mat out = imageProcessor.getOutputImage();



		timer.stop();
		cout << timer.getElapsedTimeInMicroSec() << " ";
		totalTime += timer.getElapsedTimeInMicroSec();

		cv::imshow("Title", out);
	}
	cout << " Avg: " << totalTime / 1 << "us\n";

	cv::waitKey(0);
}

void CannyDetailCPUTest(size_t size)
{
	// create random image
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(64, 25);
	unsigned char *pRandomImage = (unsigned char *)malloc(size * size);

	for (size_t pixel = 0; pixel < size * size; pixel++)
	{
		pRandomImage[pixel] = (unsigned char)std::round(d(gen));
	}

	Mat inputImage(size, size, CV_8UC1, pRandomImage);
	Mat out;

	Timer timer;

	cout << "Size: " << size << "\n";

	timer.start();
	CPUCanny imageProcessor;
	timer.stop();
	cout << "Allocator: " << timer.getElapsedTimeInMicroSec() << "\n";

	timer.start();
	imageProcessor.LoadOCVImage(inputImage);
	timer.stop();
	cout << "LoadImage: " << timer.getElapsedTimeInMicroSec() << "\n";


	timer.start();
	imageProcessor.Gaussian();
	timer.stop();
	cout << "Gaussian: " << timer.getElapsedTimeInMicroSec() << "\n";

	timer.start();
	imageProcessor.Sobel();
	timer.stop();
	cout << "Sobel: " << timer.getElapsedTimeInMicroSec() << "\n";

	timer.start();
	imageProcessor.NonMaximaSuppression();
	timer.stop();
	cout << "NMS: " << timer.getElapsedTimeInMicroSec() << "\n";

	timer.start();
	imageProcessor.HysteresisThresholding();
	timer.stop();
	cout << "Hysteresis: " << timer.getElapsedTimeInMicroSec() << "\n";
}

void CannyDetailGPUTest(size_t size)
{
	// create random image
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(64, 25);
	unsigned char *pRandomImage = (unsigned char *)malloc(size * size);

	for (size_t pixel = 0; pixel < size * size; pixel++)
	{
		pRandomImage[pixel] = (unsigned char)std::round(d(gen));
	}

	Mat inputImage(size, size, CV_8UC1, pRandomImage);
	Mat out;

	Timer timer;

	cout << "Size: " << size << "\n";

	timer.start();
	OCLCanny imageProcessor;
	timer.stop();
	cout << "Allocator: " << timer.getElapsedTimeInMicroSec() << "\n";

	timer.start();
	imageProcessor.LoadOCVImage(inputImage);
	timer.stop();
	cout << "LoadImage: " << timer.getElapsedTimeInMicroSec() << "\n";


	timer.start();
	imageProcessor.Gaussian();
	imageProcessor.wait();
	timer.stop();
	cout << "Gaussian: " << timer.getElapsedTimeInMicroSec() << "\n";

	timer.start();
	imageProcessor.Sobel();
	imageProcessor.wait();
	timer.stop();
	cout << "Sobel: " << timer.getElapsedTimeInMicroSec() << "\n";

	timer.start();
	imageProcessor.NonMaximaSuppression();
	imageProcessor.wait();
	timer.stop();
	cout << "NMS: " << timer.getElapsedTimeInMicroSec() << "\n";

	timer.start();
	imageProcessor.HysteresisThresholding();
	imageProcessor.wait();
	timer.stop();

	cout << "Hysteresis: " << timer.getElapsedTimeInMicroSec() << "\n";
}

void CannyRealImageTest()
{
#define DEBUG_PRINT
	Mat rawImage = cv::imread("D:\\image_samples\\machine.jpg");
	Mat input;
	cv::cvtColor(rawImage, input, cv::COLOR_BGR2GRAY);

	OCLCanny imageProcessor;
	imageProcessor.LoadOCVImage(input);

	imageProcessor.Gaussian();
	imageProcessor.Sobel();
	imageProcessor.NonMaximaSuppression();
	imageProcessor.HysteresisThresholding();

	Mat output = imageProcessor.getOutputImage();
	cv::imshow("Title2", input);
	cv::imshow("Title", output);
	cv::waitKey(0);
}

int main(int argc, char **argv)
{

	
	CannyDetailGPUTest(int(pow(2, 12)));
	CannyDetailCPUTest(int(pow(2, 12)));

	return 0;
}