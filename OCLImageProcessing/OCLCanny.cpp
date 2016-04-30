#include "OCLCanny.h"
#include "utils.h"
#include <iostream>
#include <fstream>

using std::string;
using std::ifstream;
using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using cv::Mat;
using cv::UMat;


OCLCanny::OCLCanny(bool type)
{
	// Initialize OCL
	try
	{
		// find all available platforms
		cl::Platform::get(&allPlatforms);

		if (type == USING_GPU)
		{
			allPlatforms[0].getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
		}
		else if (type == USING_CPU)
		{
			allPlatforms[0].getDevices(CL_DEVICE_TYPE_CPU, &allDevices);
		}

		targetDevice = selectDevice(0);

		// create OCL context
		context = cl::Context(allDevices);

		// create OCL command queue
		queue = cl::CommandQueue(context, targetDevice);

		// create and load kernels
		gaussianBlurKernel = LoadKernel("canny.cl", "gaussian_blur");
		// sobelOperatorKernel = LoadKernel("canny.cl", "sobel_operation");
		// nonMaximaSuppressionKernel = LoadKernel("canny.cl", "non_maxima_suppression");
		// hysteresisThresholdingKernel = LoadKernel("canny.cl", "hysteresis_thresholding");

	}
	catch (const exception &e)
	{
		cerr << "Error: " << e.what() << ": " << endl;
	}

	// print all available ocl platforms
	std::cout << "OpenCL Platforms:\n";
	for (int idx = 0; idx < allPlatforms.size(); idx++)
	{
		std::cout << "[";
		std::cout << (idx == 0 ? "*]" : " ]");
		std::cout << allPlatforms[idx].getInfo<CL_PLATFORM_VENDOR>() << std::endl;

	}
}

void OCLCanny::LoadImage(Mat &rawImage)
{
	// crop image from rows and cols that to be the integer multiple of groupsize
	// after substracting 2 from them

	int rows = ((rawImage.rows - 2) / workgroup_size) * workgroup_size + 2;
	int cols = ((rawImage.cols - 2) / workgroup_size) * workgroup_size + 2;
	cv::Rect croppedArea(0, 0, cols, rows);

	// read image
	inputBuffer = rawImage(croppedArea).clone();
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

void OCLCanny::LoadImage(UMat &rawImage)
{
	Mat img;
	rawImage.copyTo(img);
	LoadImage(img);
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
			cl::NDRange(inputBuffer.rows, inputBuffer.cols),
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
}

void OCLCanny::NonMaximaSuppression()
{
}

void OCLCanny::HysteresisThresholding()
{
}

OCLCanny::~OCLCanny()
{
}

cl::Device & OCLCanny::selectDevice(int idx)
{
	if (allDevices.size() == 0)
	{
		throw(string("No OCL devices"));
	}

	if (allDevices.size() <= idx)
	{
		throw(string("Select device out of range"));
	}

	return allDevices[idx];
}

cl::Kernel OCLCanny::LoadKernel(string kernelFileName, string kernelName)
{
	// Read from kernel file and create program
	string oclString = FileToString(kernelFileName);
	cl::Program::Sources sources(1, std::make_pair(oclString.c_str(), oclString.length()));
	cl::Program program(context, sources);

	// use jit compiler to build program for all available targets
	try {
		program.build(allDevices);
	} catch (...)
	{
		// print build log
		cerr << "Build Status:\n"
			<< program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(targetDevice)
			<< endl << "Build Options:\n"
			<< program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(targetDevice)
			<< endl << "Build Log:\n"
			<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(targetDevice) << endl;
	}

	return cl::Kernel(program, kernelName.c_str());
}


