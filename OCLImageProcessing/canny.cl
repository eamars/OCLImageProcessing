__constant float gaussian_kernel[5][5] = {
	{ 0.00224214, 0.0165673, 0.0165673, 0.0165673, 0.00224214 },
	{ 0.0165673, 0.0450347, 0.122417, 0.0450347, 0.0165673 },
	{ 0.0165673, 0.122417, 0.122417, 0.122417, 0.0165673 },
	{ 0.0165673, 0.0450347, 0.122417, 0.0450347, 0.0165673 },
	{ 0.00224214, 0.0165673, 0.0165673, 0.0165673, 0.00224214 }
};

__constant int sobel_gx_kernel[3][3] = {
	{ -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, 1 }
};

__constant int sobel_gy_kernel[3][3] = {
	{ -1,-2,-1 },
	{ 0, 0, 0 },
	{ 1, 2, 1 }
};

__kernel void gaussian_blur(
	__global uchar *inImage,
	__global uchar *outImage,
	size_t rows, size_t cols)
{
	int sum = 0;
	size_t row = get_global_id(0);
	size_t col = get_global_id(1);
	size_t pos = row * cols + col;

	for (int i = 0; i < 5; i++)
		#pragma unroll
		for (int j = 0; j < 5; j++)
			sum += gaussian_kernel[i][j] * inImage[(i + row - 1)*cols + (j + col - 1)];

	outImage[pos] = min(255, max(0, sum));
}

__kernel void sobel_operation(
	__global uchar *inImage,
	__global uchar *outImage,
	__global uchar *theta,
	size_t rows, size_t cols)
{
	const float MPI = 3.14159265f;
	float sumx = 0, sumy = 0, angle = 0;
	size_t row = get_global_id(0);
	size_t col = get_global_id(1);
	size_t pos = row * cols + col;

	// find gx and gy
	for (int i = 0; i < 3; i++)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			sumx += sobel_gx_kernel[i][j] * inImage[(i + row - 1) * cols + (j + col - 1)];
			sumy += sobel_gy_kernel[i][j] * inImage[(i + row - 1) * cols + (j + col - 1)];
		}
	}

	// hypot is defined as sqrt(x^2, y^2)
	outImage[pos] = min(255, max(0, (int)hypot(sumx, sumy)));

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

__kernel void non_maxima_suppression(
	__global uchar *inImage,
	__global uchar *outImage,
	__global uchar *theta,
	size_t rows,
	size_t cols
)
{
	size_t row = get_global_id(0);
	size_t col = get_global_id(1);

	// define directions
	const size_t pos = row * cols + col;
	const size_t N = (row - 1) * cols + col;
	const size_t NE = (row - 1) * cols + (col + 1);
	const size_t E = row * cols + (col + 1);
	const size_t SE = (row + 1) * cols + (col + 1);
	const size_t S = (row + 1) * cols + col;
	const size_t SW = (row + 1) * cols + (col - 1);
	const size_t W = row * cols + (col - 1);
	const size_t NW = (row - 1) * cols + (col - 1);

	switch (theta[pos])
	{
		case 0:
		{
			// supress current pixel if neighbour has larger magnitude
			if (inImage[pos] <= inImage[E] || inImage[pos] <= inImage[W])
			{
				outImage[pos] = 0;
			}

			// otherwise use current value
			else
			{
				outImage[pos] = inImage[pos];
			}
			break;
		}

		case 45:
		{
			if (inImage[pos] <= inImage[NE] || inImage[pos] <= inImage[SW])
			{
				outImage[pos] = 0;
			}
			else
			{
				outImage[pos] = inImage[pos];
			}
			break;
		}

		case 90:
		{
			if (inImage[pos] <= inImage[N] || inImage[pos] <= inImage[S])
			{
				outImage[pos] = 0;
			}
			else
			{
				outImage[pos] = inImage[pos];
			}
			break;
		}

		case 135:
		{
			if (inImage[pos] <= inImage[NW] || inImage[pos] <= inImage[SE])
			{
				outImage[pos] = 0;
			}
			else
			{
				outImage[pos] = inImage[pos];
			}
			break;
		}

		default:
		{
			outImage[pos] = inImage[pos];
			break;
		}
	}
}

__kernel void hysteresis_thresholding(
	__global uchar *inImage,
	__global uchar *outImage,
	size_t rows, size_t cols
)
{
	float low = 10.0f;
	float high = 70.0f;
	float median = (low + high) / 2;
	const uchar EDGE = 255;

	size_t row = get_global_id(0);
	size_t col = get_global_id(1);
	size_t pos = row * cols + col;

	// in each position of (x, y), output the pixel if it is strong
	if (inImage[pos] >= high)
	{
		outImage[pos] = EDGE;
	}

	// discard the pixel (x, y) if it is weak
	else if (inImage[pos] <= low)
	{
		outImage[pos] = 0;
	}

	// if the pixel is a candidate, 
	else
	{
		if (inImage[pos] >= median)
		{
			outImage[pos] = EDGE;
		}
		else
		{
			outImage[pos] = 0;
		}
	}
}