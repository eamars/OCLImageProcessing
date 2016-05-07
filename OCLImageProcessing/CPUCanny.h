#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class CPUCanny
{
private:
	unsigned char *gaussian;
	unsigned char *sobel;
	unsigned char *nonmaxima;
	unsigned char *hysteresis;
	unsigned char *theta;

	int buffer_idx;

	cv::Mat inputBuffer;

public:
	CPUCanny();
	~CPUCanny();

	void LoadOCVImage(cv::Mat & rawImage);
	cv::Mat Gaussian();
	cv::Mat Sobel();
	cv::Mat NonMaximaSuppression();
	cv::Mat HysteresisThresholding();
};

