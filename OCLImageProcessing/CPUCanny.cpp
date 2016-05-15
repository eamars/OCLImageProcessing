#include "CPUCanny.h"
#include <iostream>
#include <algorithm>
#include <stack>
#include <tuple>

using std::min;
using std::max;
using std::make_tuple;
using std::get;
using cv::Mat;
using std::stack;
using std::tuple;
using cv::Scalar;
using cv::Vec3b;

CPUCanny::CPUCanny()
{
}


CPUCanny::~CPUCanny()
{
	free(gaussian);
	free(sobel);
	free(nonmaxima);
	free(hysteresis);
	free(theta);
}

void CPUCanny::LoadOCVImage(cv::Mat & rawImage)
{
	inputBuffer = rawImage.clone();
}

Mat CPUCanny::Gaussian()
{
	const float gaussian_kernel[5][5] = {
		{ 0.00224214, 0.0165673, 0.0165673, 0.0165673, 0.00224214 },
		{ 0.0165673, 0.0450347, 0.122417, 0.0450347, 0.0165673 },
		{ 0.0165673, 0.122417, 0.122417, 0.122417, 0.0165673 },
		{ 0.0165673, 0.0450347, 0.122417, 0.0450347, 0.0165673 },
		{ 0.00224214, 0.0165673, 0.0165673, 0.0165673, 0.00224214 }
	};

	gaussian = (unsigned char *)malloc(inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());

	// image
	for (int row = 3; row < inputBuffer.rows - 3; row++)
	{
		for (int col = 3; col < inputBuffer.cols - 3; col++)
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

			gaussian[pos] = min(255, max(0, sum));
		}
	}
	return Mat(inputBuffer.rows, inputBuffer.cols, CV_8UC1, gaussian);
}

cv::Mat CPUCanny::Sobel()
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

	sobel = (unsigned char *)malloc(inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());
	theta = (unsigned char *)malloc(inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());

	// image
	for (int row = 1; row < inputBuffer.rows - 1; row++)
	{
		for (int col = 1; col < inputBuffer.cols - 1; col++)
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
					sumx += sobel_gx_kernel[i][j] * gaussian[idx];
					sumy += sobel_gy_kernel[i][j] * gaussian[idx];
				}
			}

			sobel[pos] = min(255, max(0, (int)hypot(sumx, sumy)));

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
					theta[pos] = 0;
				}
				else if (angle <= 3 * MPI / 8)
				{
					theta[pos] = 45;
				}
				else if (angle <= 5 * MPI / 8)
				{
					theta[pos] = 90;
				}
				else if (angle <= 7 * MPI / 8)
				{
					theta[pos] = 135;
				}
				else
				{
					theta[pos] = 0;
				}
			}
			else
			{
				if (angle <= 9 * MPI / 8)
				{
					theta[pos] = 0;
				}
				else if (angle <= 11 * MPI / 8)
				{
					theta[pos] = 45;
				}
				else if (angle <= 13 * MPI / 8)
				{
					theta[pos] = 90;
				}
				else if (angle <= 15 * MPI / 8)
				{
					theta[pos] = 135;
				}
				else
				{
					theta[pos] = 0;
				}
			}
		}
	}
	return Mat(inputBuffer.rows, inputBuffer.cols, CV_8UC1, sobel);
}

cv::Mat CPUCanny::NonMaximaSuppression()
{
	nonmaxima = (unsigned char *)malloc(inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());

	for (int row = 1; row < inputBuffer.rows - 1; row++)
	{
		for (int col = 1; col < inputBuffer.cols - 1; col++)
		{
			// define directions
			const int pos = row * inputBuffer.cols + col;
			const int N = (row - 1) * inputBuffer.cols + col;
			const int NE = (row - 1) * inputBuffer.cols + (col + 1);
			const int E = row * inputBuffer.cols + (col + 1);
			const int SE = (row + 1) * inputBuffer.cols + (col + 1);
			const int S = (row + 1) * inputBuffer.cols + col;
			const int SW = (row + 1) * inputBuffer.cols + (col - 1);
			const int W = row * inputBuffer.cols + (col - 1);
			const int NW = (row - 1) * inputBuffer.cols + (col - 1);

			
			switch (theta[pos])
			{
				case 0:
				{
					// supress current pixel if neighbour has larger magnitude
					if (sobel[pos] < sobel[E] || sobel[pos] < sobel[W])
					{
						nonmaxima[pos] = 0;
					}

					// otherwise use current value
					else
					{
						nonmaxima[pos] = sobel[pos];
					}
					break;
				}

				case 45:
				{
					if (sobel[pos] < sobel[NE] || sobel[pos] < sobel[SW])
					{
						nonmaxima[pos] = 0;
					}
					else
					{
						nonmaxima[pos] = sobel[pos];
					}
					break;
				}

				case 90:
				{
					if (sobel[pos] < sobel[N] || sobel[pos] < sobel[S])
					{
						nonmaxima[pos] = 0;
					}
					else
					{
						nonmaxima[pos] = sobel[pos];
					}
					break;
				}

				case 135:
				{
					if (sobel[pos] < sobel[NW] || sobel[pos] < sobel[SE])
					{
						nonmaxima[pos] = 0;
					}
					else
					{
						nonmaxima[pos] = sobel[pos];
					}
					break;
				}

				default:
				{
					nonmaxima[pos] = sobel[pos];
					break;
				}
			}
		}
	}

	return Mat(inputBuffer.rows, inputBuffer.cols, CV_8UC1, nonmaxima);
}


void print_data(int rows, int cols, unsigned char *in)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%03u ", in[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}

const int move_dir[2][8] = {
	{ -1, -1, -1, 0, 0, 1, 1, 1 },
	{ -1, 0, 1, -1, 1, -1, 0, 1 }
};


void traceRecursive(unsigned char *in, unsigned char *out, int row, int col, int rows, int cols, int tLow)
{
	// in[row][col] is edge, then find adjacent pixels

	for (int i = 0; i < 8; i++)
	{
		int x = col + move_dir[0][i];
		int y = row + move_dir[1][i];

		if (x >= 0 && x < cols && y >= 0 && y < rows)
		{
			int pos = y * cols + x;

			// adjacent pixels
			if (in[pos] >= tLow && out[pos] != 255)
			{
				out[pos] = 255;
				traceRecursive(in, out, y, x, rows, cols, tLow);
			}
		}
		
	}
}

void traceStack(unsigned char *in, unsigned char *out, int row, int col, int rows, int cols, int tLow)
{
	stack<tuple<int, int>> edges;

	for (int i = 0; i < 8; i++)
	{
		int x = col + move_dir[0][i];
		int y = row + move_dir[1][i];

		if (x >= 0 && x < cols && y >= 0 && y < rows)
		{
			int pos = y * cols + x;

			// adjacent pixels
			if (in[pos] >= tLow && out[pos] != 255)
			{
				out[pos] = 255;
				edges.push(make_tuple(x, y));
			}
		}
	}

	while (!edges.empty())
	{
		tuple<int, int> pair = edges.top();
		edges.pop();

		for (int i = 0; i < 8; i++)
		{
			int x = get<0>(pair) + move_dir[0][i];
			int y = get<1>(pair) + move_dir[1][i];

			if (x >= 0 && x < cols && y >= 0 && y < rows)
			{
				int pos = y * cols + x;

				// adjacent pixels
				if (in[pos] >= tLow && out[pos] != 255)
				{
					out[pos] = 255;
					edges.push(make_tuple(x, y));
				}
			}
		}
	}
}

cv::Mat CPUCanny::HysteresisThresholding()
{
	hysteresis = (unsigned char *)malloc(inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());
	
	// reset all output to low
	memset(hysteresis, 0x00, inputBuffer.rows * inputBuffer.cols * inputBuffer.elemSize());

	unsigned char *in = nonmaxima;
	unsigned char *out = hysteresis;
	int rows = inputBuffer.rows;
	int cols = inputBuffer.cols;
	
	const unsigned char tHigh = 80;
	const unsigned char tLow = 50;
	
	for (int row = 1; row < rows - 1; row++)
	{
		for (int col = 1; col < cols - 1; col++)
		{
			const int pos = row * inputBuffer.cols + col;
			if (in[pos] > tHigh && out[pos] != 255)
			{
				out[pos] = 255;
				traceStack(in, out, row, col, rows, cols, tLow);
			}

		}
	}


	return Mat(inputBuffer.rows, inputBuffer.cols, CV_8UC1, out);
}

Mat CPUCanny::getTheta()
{
	if (theta == NULL)
	{
		return Mat();
	}

	// create new empty image
	Mat RGBTheta(inputBuffer.rows, inputBuffer.cols, CV_8UC3, Scalar(0, 0, 0));

	// fill with direction data
	for (int row = 0; row < inputBuffer.rows; row++)
	{
		for (int col = 0; col < inputBuffer.cols; col++)
		{
			int pos = row * inputBuffer.cols + col;
			switch (theta[pos])
			{
				case 0:
				{
					break;
				}
				case 45:
				{
					RGBTheta.at<Vec3b>(row, col) = Vec3b(0, 255, 0);
					break;
				}
				case 90:
				{
					RGBTheta.at<Vec3b>(row, col) = Vec3b(255, 0, 0);
					break;
				}
				case 135:
				{
					RGBTheta.at<Vec3b>(row, col) = Vec3b(0, 0, 255);
					break;
				}
				default:
				{
					break;
				}
			}
		}
	}
	return RGBTheta;
}

