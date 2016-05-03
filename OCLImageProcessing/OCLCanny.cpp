#include "OCLCanny.h"
#include "utils.h"
#include "cpu.h"
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
			setWorkgroupSize(16);
		}
		else if (type == USING_CPU)
		{
			allPlatforms[0].getDevices(CL_DEVICE_TYPE_CPU, &allDevices);
			setWorkgroupSize(1);
		}

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

Mat OCLCanny::GaussianWithCPU()
{
	const float gaussian_kernel[5][5] = {
		{ 0.00224214, 0.0165673, 0.0165673, 0.0165673, 0.00224214 },
		{ 0.0165673, 0.0450347, 0.122417, 0.0450347, 0.0165673 },
		{ 0.0165673, 0.122417, 0.122417, 0.122417, 0.0165673 },
		{ 0.0165673, 0.0450347, 0.122417, 0.0450347, 0.0165673 },
		{ 0.00224214, 0.0165673, 0.0165673, 0.0165673, 0.00224214 }
	};

	unsigned char *pOutputImage = (unsigned char *)malloc(inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());

	// image
	for (int row = 1; row < inputBuffer.rows; row++)
	{
		for (int col = 1; col < inputBuffer.cols; col++)
		{
			int pos = row * inputBuffer.cols + col;
			int sum = 0;

			// kernel
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					int idx = (i + row - 1)*inputBuffer.cols + (j + col - 1);
					sum += gaussian_kernel[i][j] * inputBuffer.data[idx];
				}
			}

			pOutputImage[pos] = min(255, max(0, sum));
		}
	}
	return Mat(inputBuffer.rows, inputBuffer.cols, CV_8UC1, pOutputImage);
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

cv::Mat OCLCanny::SobelWithCPU(Mat & theta)
{
	const float MPI = 3.14159265f;
	const int sobel_gx_kernel[3][3] = {
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 }
	};

	const int sobel_gy_kernel[3][3] = {
		{ -1,-2,-1 },
		{ 0, 0, 0 },
		{ 1, 2, 1 }
	};

	unsigned char *pOutputImage = (unsigned char *)malloc(inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());

	// image
	for (int row = 1; row < inputBuffer.rows; row++)
	{
		for (int col = 1; col < inputBuffer.cols; col++)
		{
			float sumx = 0, sumy = 0, angle = 0;
			int pos = row * inputBuffer.cols + col;
			int sum = 0;

			// kernel
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					int idx = (i + row - 1)*inputBuffer.cols + (j + col - 1);
					sumx += sobel_gx_kernel[i][j] * inputBuffer.data[idx];
					sumy += sobel_gy_kernel[i][j] * inputBuffer.data[idx];
				}
			}

			pOutputImage[pos] = min(255, max(0, (int)hypot(sumx, sumy)));

			// get direction
			angle = atan2(sumy, sumx);

			// if angle is negative, then shift by 2PI
			if (angle < 0.0f)
			{
				angle = fmod((angle + 2 * MPI), (2 * MPI));
			}

			// round angles to 0, 45, 90 and 135 degs
			// angles are equally likely to distribute between 
			// 0~PI and PI~2PI
			if (angle <= MPI)
			{
				if (angle <= MPI / 8)
				{
					theta.data[pos] = 0;
				}
				else if (angle <= 3 * MPI / 8)
				{
					theta.data[pos] = 45;
				}
				else if (angle <= 5 * MPI / 8)
				{
					theta.data[pos] = 90;
				}
				else if (angle <= 7 * MPI / 8)
				{
					theta.data[pos] = 135;
				}
				else
				{
					theta.data[pos] = 0;
				}
			}
			else
			{
				if (angle <= 9 * MPI / 8)
				{
					theta.data[pos] = 0;
				}
				else if (angle <= 11 * MPI / 8)
				{
					theta.data[pos] = 45;
				}
				else if (angle <= 13 * MPI / 8)
				{
					theta.data[pos] = 90;
				}
				else if (angle <= 15 * MPI / 8)
				{
					theta.data[pos] = 135;
				}
				else
				{
					theta.data[pos] = 0;
				}
			}
		}
	}
	return Mat(inputBuffer.rows, inputBuffer.cols, CV_8UC1, pOutputImage);
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
	cout << "Building [" << kernelName << "] in [" << kernelFileName << "]"
		<< endl << "Build Status:\n"
		<< program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(targetDevice)
		<< endl << "Build Options:\n"
		<< program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(targetDevice)
		<< endl << "Build Log:\n"
		<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(targetDevice) << endl;

	return cl::Kernel(program, kernelName.c_str());
}


