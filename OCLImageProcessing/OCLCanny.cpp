#include "OCLCanny.h"
#include "utils.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

using std::string;
using std::ifstream;
using std::min;
using std::max;
using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using cv::Mat;
using cv::UMat;


OCLCanny::OCLCanny()
{
	// Initialize OCL
	try
	{
		// find all available platforms
		cl::Platform::get(&allPlatforms);

		allPlatforms[0].getDevices(CL_DEVICE_TYPE_GPU, &allDevices);

		targetDevice = allDevices[0];

		// create OCL context
		context = cl::Context(allDevices);

		// create OCL command queue
		queue = cl::CommandQueue(context, targetDevice);

		// create and load kernels
		gaussianBlurKernel = LoadKernel("canny.cl", "gaussian_blur");
		sobelOperatorKernel = LoadKernel("canny.cl", "sobel_operation");
		nonMaximaSuppressionKernel = LoadKernel("canny.cl", "non_maxima_suppression");
		hysteresisThresholdingKernel = LoadKernel("canny.cl", "hysteresis_thresholding");

	}
	catch (const exception &e)
	{
		cerr << "Error: " << e.what() << ": " << endl;
	}

	// print all available ocl platforms
#ifdef DEBUG_PRINT
	std::cout << "OpenCL Platforms:\n";
	for (int idx = 0; idx < allPlatforms.size(); idx++)
	{
		std::cout << "[";
		std::cout << (idx == 0 ? "*]" : " ]");
		std::cout << allPlatforms[idx].getInfo<CL_PLATFORM_VENDOR>() << std::endl;

	}
#endif
}

void OCLCanny::LoadOCVImage(Mat &rawImage)
{/*
	int rows = ((rawImage.rows - 2) / workgroup_size) * workgroup_size + 2;
	int cols = ((rawImage.cols - 2) / workgroup_size) * workgroup_size + 2;
	cv::Rect croppedArea(0, 0, cols, rows);
	*/
	// read image
	inputBuffer = rawImage.clone();
	outputBuffer = Mat(inputBuffer.rows, inputBuffer.cols, CV_8UC1);

	// setup buffers
	NextBuffer() = cl::Buffer(
		context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize(),
		inputBuffer.data);

	PrevBuffer() = cl::Buffer(
		context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());

	theta = cl::Buffer(
		context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());

	SwapBuffer();
}


cv::Mat OCLCanny::getOutputImage()
{
	queue.enqueueReadBuffer(
		PrevBuffer(),
		CL_TRUE,
		0,
		inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize(),
		outputBuffer.data);

	wait();

	assert(outputBuffer.rows == inputBuffer.rows && outputBuffer.cols == inputBuffer.cols);
	return outputBuffer;
}

void OCLCanny::wait()
{
	queue.finish();
}

void OCLCanny::setWorkgroupSize(int size)
{
	workgroup_size = size;
}

void OCLCanny::Gaussian()
{
	try
	{
		// set arguments
		gaussianBlurKernel.setArg(0, PrevBuffer());
		gaussianBlurKernel.setArg(1, NextBuffer());
		gaussianBlurKernel.setArg(2, (size_t)inputBuffer.rows);
		gaussianBlurKernel.setArg(3, (size_t)inputBuffer.cols);

		// enqueue
		queue.enqueueNDRangeKernel(
			gaussianBlurKernel,
			cl::NDRange(1, 1),
			cl::NDRange(inputBuffer.rows - 2, inputBuffer.cols - 2),
			cl::NDRange(workgroup_size, workgroup_size),
			NULL
		);
		
	}
	catch (const exception &e)
	{
		cerr << "Error: " << e.what() << endl;
	}

	// swap the input and ouput
	SwapBuffer();
}

void OCLCanny::Sobel()
{
	sobelOperatorKernel.setArg(0, PrevBuffer());
	sobelOperatorKernel.setArg(1, NextBuffer());
	sobelOperatorKernel.setArg(2, theta);
	sobelOperatorKernel.setArg(3, (size_t)inputBuffer.rows);
	sobelOperatorKernel.setArg(4, (size_t)inputBuffer.cols);

	queue.enqueueNDRangeKernel(
		sobelOperatorKernel,
		cl::NDRange(1, 1),
		cl::NDRange(inputBuffer.rows - 2, inputBuffer.cols - 2),
		cl::NDRange(workgroup_size, workgroup_size),
		NULL
	);

	SwapBuffer();
}

void OCLCanny::NonMaximaSuppression()
{
	nonMaximaSuppressionKernel.setArg(0, PrevBuffer());
	nonMaximaSuppressionKernel.setArg(1, NextBuffer());
	nonMaximaSuppressionKernel.setArg(2, theta);
	nonMaximaSuppressionKernel.setArg(3, (size_t)inputBuffer.rows);
	nonMaximaSuppressionKernel.setArg(4, (size_t)inputBuffer.cols);

	queue.enqueueNDRangeKernel(
		nonMaximaSuppressionKernel,
		cl::NDRange(1, 1),
		cl::NDRange(inputBuffer.rows - 2, inputBuffer.cols - 2),
		cl::NDRange(workgroup_size, workgroup_size),
		NULL
	);

	SwapBuffer();
}

void OCLCanny::HysteresisThresholding()
{
	hysteresisThresholdingKernel.setArg(0, PrevBuffer());
	hysteresisThresholdingKernel.setArg(1, NextBuffer());
	hysteresisThresholdingKernel.setArg(2, (size_t)inputBuffer.rows);
	hysteresisThresholdingKernel.setArg(3, (size_t)inputBuffer.cols);

	queue.enqueueNDRangeKernel(
		hysteresisThresholdingKernel,
		cl::NDRange(1, 1),
		cl::NDRange(inputBuffer.rows - 2, inputBuffer.cols - 2),
		cl::NDRange(workgroup_size, workgroup_size),
		NULL
	);

	SwapBuffer();
}

OCLCanny::~OCLCanny()
{
}

cl::Kernel OCLCanny::LoadKernel(string kernelFileName, string kernelName)
{
	// Read from kernel file and create program
	string oclString = FileToString(kernelFileName);
	cl::Program::Sources sources(1, std::make_pair(oclString.c_str(), oclString.length()));
	cl::Program program(context, sources);

	// use jit compiler to build program for all available targets
	program.build(allDevices);

	// print build log
#ifdef DEBUG_PRINT
	cout << "Building [" << kernelName << "] in [" << kernelFileName << "]"
		<< endl << "Build Status:\n"
		<< program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(targetDevice)
		<< endl << "Build Options:\n"
		<< program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(targetDevice)
		<< endl << "Build Log:\n"
		<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(targetDevice) << endl;
#endif

	return cl::Kernel(program, kernelName.c_str());
}


