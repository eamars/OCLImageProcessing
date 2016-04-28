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


OCLCanny::OCLCanny(Mat & rawImage, bool type)
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
		context = cl::Context(allDevices);
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
}

OCLCanny::OCLCanny(UMat & rawImage, bool type)
{
	Mat img;
	rawImage.copyTo(img);
	OCLCanny(img, type);
}

cv::Mat OCLCanny::getOutputImage()
{
	// dummy function
	return cv::Mat();
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

void OCLCanny::Canny()
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


