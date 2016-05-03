#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#include "ocl.h"
#include "cpu.h"
#include "gpu.h"
#include "utils.h"
#include "Timer.h"

#include "OCLCanny.h"

using std::cerr;
using std::cout;
using std::string;
using std::vector;
using std::endl;
using cv::Mat;
using cv::Size;

void CPUTest(Mat &grayImage, Timer &timer)
{
	// get float data
	unsigned char *pImage = grayImage.data;
	Size sz = grayImage.size();
	size_t size = sz.width * sz.height;

	unsigned char *pNewImage = (unsigned char *)malloc(size);

	// create gaussian filter
	float kernel[25];
#ifdef _DEBUG
	cout << "[CPU] Time taken for creating Gaussian filter mask: ";
#endif
	timer.start();
	createGaussianFilter(kernel, 5, 1.0f);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// gaussian blur
#ifdef _DEBUG
	cout << "[CPU] Time taken for performing matrix convolution: ";
#endif
	timer.start();
	CPUConvolve(pImage, pNewImage, sz.width, sz.height, kernel, 5, 5);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// comparison 1
#ifdef _DEBUG
	cout << "[CPU] Time taken for performing matrix convolution (FAST): ";
#endif
	timer.start();
	CPUConvolveFast(pImage, pNewImage, sz.width, sz.height, kernel, 5, 5);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	free(pNewImage);
}


void GPUTest(Mat &grayImage, Timer &timer)
{
	// get float data
	unsigned char *pImage = grayImage.data;
	Size sz = grayImage.size();
	size_t size = sz.width * sz.height;

	unsigned char *pGaussianImage = (unsigned char *)malloc(size);

	// create gaussian filter
	float gaussianFilter[25];
#ifdef _DEBUG
	cout << "[GPU] Time taken for creating Gaussian filter mask: ";
#endif
	timer.start();
	createGaussianFilter(gaussianFilter, 5, 1.0f);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	for (int i = 0; i < 25; i++)
	{
		cout << gaussianFilter[i] << ", ";
	}

	// connect OCL device and load scripts
	oclHandle handle;

#ifdef _DEBUG
	cout << "[GPU] Time taken for connecting to OCL devices: ";
#endif
	vector<string> kernelNames;
	kernelNames.push_back(string("Convolve"));
	try
	{
		timer.start();
		CLInit(&handle, string("convolve.cl"), kernelNames);
		timer.stop();
#ifdef _DEBUG
		cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif
	}
	catch (string errmsg)
	{
		cout << errmsg << endl;
	}
	
	// copy memory content
	oclBuffer oclBuffers;
	hostBuffer hostBuffers;

	hostBuffers.in = pImage;
	hostBuffers.out = pGaussianImage;
	hostBuffers.dataSizeX = sz.width;
	hostBuffers.dataSizeY = sz.height;

	hostBuffers.kernel = gaussianFilter;
	hostBuffers.kernelSizeX = 5;
	hostBuffers.kernelSizeY = 5;

#ifdef _DEBUG
	cout << "[GPU] Time taken for copying memory content: ";
#endif
	timer.start();
	CLBufferInit(&handle, &oclBuffers, &hostBuffers);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif
	
	// enqueue kernels
#ifdef _DEBUG
	cout << "[GPU] Time taken for performing matrix convolution (OCL): ";
#endif
	timer.start();
	GPUConvolve(&handle, &oclBuffers, 0);
	timer.stop();
#ifdef _DEBUG
		cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// create new container for sobel operators
	unsigned char *pSobelGx = (unsigned char *)malloc(size);
	unsigned char *pSobelGy = (unsigned char *)malloc(size);

	// create sobel operator
	float kernelSobelGx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float kernelSobelGy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	// sobel gx
	hostBuffers.in = pGaussianImage;
	hostBuffers.out = pSobelGx;
	hostBuffers.dataSizeX = sz.width;
	hostBuffers.dataSizeY = sz.height;

	hostBuffers.kernel = kernelSobelGx;
	hostBuffers.kernelSizeX = 3;
	hostBuffers.kernelSizeY = 3;

	CLBufferInit(&handle, &oclBuffers, &hostBuffers);
	GPUConvolve(&handle, &oclBuffers, 0);

	// sobel gy
	hostBuffers.in = pSobelGx;
	hostBuffers.out = pSobelGy;
	hostBuffers.dataSizeX = sz.width;
	hostBuffers.dataSizeY = sz.height;

	hostBuffers.kernel = kernelSobelGy;
	hostBuffers.kernelSizeX = 3;
	hostBuffers.kernelSizeY = 3;

	CLBufferInit(&handle, &oclBuffers, &hostBuffers);
	GPUConvolve(&handle, &oclBuffers, 0);
	
	// show the example pSobelGx
	Mat gaussianImage = Mat(sz, CV_8UC1, pGaussianImage);
	Mat sobelImage = Mat(sz, CV_8UC1, pSobelGy);
	cv::imshow("Gray Image", grayImage);
	cv::imshow("Blur Image", gaussianImage);
	cv::imshow("Sobel Image", sobelImage);
	cv::waitKey(0);
	
	// release buffer
	free(pGaussianImage);
	free(pSobelGx);
	free(pSobelGy);
}

void realImageTest(int argc, char **argv)
{
	// check for parameters
	if (argc < 2)
	{
		cerr << "Insufficient parameters" << endl;
		exit(-1);
	}

	Mat grayImage;
	Mat rawImage;

	// set timer
	Timer timer;

	// read raw image
#ifdef _DEBUG
	cout << "Time taken for loading image: ";
#endif
	timer.start();
	rawImage = cv::imread(argv[1]);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// conver to gray scale
#ifdef _DEBUG
	cout << "Time taken for converting to gray scale: ";
#endif
	timer.start();
	cv::cvtColor(rawImage, grayImage, cv::COLOR_BGR2GRAY);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// reference test 1 by OpenCV
	Mat ocvNewImage;
#ifdef _DEBUG
	cout << "Time taken for performing matrix convolution (OCV): ";
#endif
	timer.start();
	cv::GaussianBlur(grayImage, ocvNewImage, cv::Size(5, 5), 1.0f);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// reference test 1 by OpenCV (using OCL)
	cv::UMat grayImageGPU = grayImage.getUMat(cv::ACCESS_READ);
	cv::UMat newImageGPU;
#ifdef _DEBUG
	cout << "Time taken for performing matrix convolution (OCV&OCL): ";
#endif
	timer.start();
	cv::GaussianBlur(grayImageGPU, newImageGPU, cv::Size(5, 5), 1.0f);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// CPU Test
	CPUTest(grayImage, timer);

	// GPU Test
	GPUTest(grayImage, timer);
}

void performanceTest(size_t size)
{
#ifdef _DEBUG
	cout << "========== Size: " << size << "==========\n";
#endif
	Timer timer;

	// create random image
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(64, 25);
	unsigned char *pRandomImage = (unsigned char *)malloc(size * size);
	
	// assign values
#ifdef _DEBUG
	cout << "Time taken for creating image: ";
#endif
	timer.start();
	for (size_t pixel = 0; pixel < size * size; pixel++)
	{
		pRandomImage[pixel] = (unsigned char)std::round(d(gen));
	}
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	Mat randomImage = Mat(size, size, CV_8UC1, pRandomImage);

	// create array for putting new image
	unsigned char *pNewImage = (unsigned char *)malloc(size * size);

	// reference test 1 by OpenCV
	Mat ocvNewImage;
#ifdef _DEBUG
	cout << "Time taken for performing matrix convolution (OCV): ";
#endif
	timer.start();
	cv::GaussianBlur(randomImage, ocvNewImage, cv::Size(5, 5), 1.0f);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// reference test 2 by OpenCV and OpenCL
	cv::UMat UmatRandomImage = randomImage.getUMat(cv::ACCESS_READ);
	cv::UMat oclNewImage;
#ifdef _DEBUG
	cout << "Time taken for performing matrix convolution (OCV&OCL): ";
#endif
	timer.start();
	cv::GaussianBlur(UmatRandomImage, oclNewImage, cv::Size(5, 5), 1.0f);
	timer.stop();
#ifdef _DEBUG
	cout << timer.getElapsedTimeInMilliSec() << " ms" << endl;
#endif

	// test 1: by CPU
	// CPUTest(randomImage, timer);

	// test 2" by GPU
	GPUTest(randomImage, timer);

	// free memory
	free(pRandomImage);
	free(pNewImage);

#ifdef _DEBUG
	cout << "===========================\n";
#endif
}

void CannyTest(size_t size)
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

	// create image in ocv format
	// Mat randomImage = Mat(size, size, CV_8UC1, pRandomImage);
	Mat randomImage, outputImage;
	Mat realImage = cv::imread("D:\\image_samples\\Lenna.png");
	cv::cvtColor(realImage, randomImage, cv::COLOR_BGR2GRAY);

	Mat gaussianImage, sobelImage, thetaImage;
	// create canny instance
	OCLCanny cannyImageProcessor(USING_GPU);
	cannyImageProcessor.LoadImage(randomImage);

	gaussianImage = cannyImageProcessor.GaussianWithCPU();
	cannyImageProcessor.Gaussian();

	sobelImage = cannyImageProcessor.SobelWithCPU(thetaImage); // TODO: allocate memory for theta!
	cannyImageProcessor.Sobel();
	// cannyImageProcessor.NonMaximaSuppression();
	// cannyImageProcessor.HysteresisThresholding();
	outputImage = cannyImageProcessor.getOutputImage();


	cv::imshow("Title", gaussianImage);
	cv::imshow("Title2", sobelImage);
	cv::waitKey(0);

	free(pRandomImage);
}

int main(int argc, char **argv)
{
	
	// realImageTest(argc, argv);
	
	/*
	for (int power = 2; power < 18; power++)
	{
		performanceTest(int(pow(2, power)));
	}
	*/
	// performanceTest(int(pow(2, 8)));
	
	CannyTest(int(pow(2, 9)));

	return 0;
}