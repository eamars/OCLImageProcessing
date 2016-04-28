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

	cl::Device &selectDevice(int idx);
	cl::Kernel LoadKernel(std::string kernelFileName, std::string kernelName);


public:
	OCLCanny(cv::Mat &rawImage, bool type=USING_GPU);
	OCLCanny(cv::UMat &rawImage, bool type=USING_GPU);
	cv::Mat getOutputImage();
	void wait();

	void setWorkgroupSize(int size);

	void Gaussian();
	void Sobel();
	void NonMaximaSuppression();
	void HysteresisThresholding();
	void Canny();

	~OCLCanny();
};

