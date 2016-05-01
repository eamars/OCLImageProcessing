#pragma once
#include <vector>
#include <string>
#include <CL/cl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#define USING_CPU false
#define USING_GPU true

class OCLCanny
{
private:
	// OCL device related
	std::vector<cl::Platform> allPlatforms;
	std::vector<cl::Device> allDevices;

	// OCL devices and queues
	cl::Platform targetPlatform;
	cl::Device targetDevice;
	cl::Context context;
	cl::CommandQueue queue;

	// OCL kernels
	cl::Kernel gaussianBlurKernel;
	cl::Kernel sobelOperatorKernel;
	cl::Kernel nonMaximaSuppressionKernel;
	cl::Kernel hysteresisThresholdingKernel;

	// workgroup size
	int workgroup_size = 1;

	cl::Kernel LoadKernel(std::string kernelFileName, std::string kernelName);

	// buffers
	int buffer_idx = 0;
	cl::Buffer buffers[2];
	cl::Buffer theta;

	// buffer operations
	inline cl::Buffer &NextBuffer()
	{
		return buffers[buffer_idx];
	}

	inline cl::Buffer &PrevBuffer()
	{
		return buffers[buffer_idx ^ 1];
	}

	inline void SwapBuffer()
	{
		buffer_idx ^= 1;
	}

	// input and output in ocv format
	cv::Mat inputBuffer, outputBuffer;


public:
	OCLCanny(bool type=USING_GPU);
	void LoadImage(cv::Mat &rawImage);
	void LoadImage(cv::UMat &rawImage);

	cv::Mat getOutputImage();

	void wait();

	void setWorkgroupSize(int size);

	void Gaussian();
	void Sobel();
	void NonMaximaSuppression();
	void HysteresisThresholding();

	~OCLCanny();
};

