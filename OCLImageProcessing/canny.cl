__constant float gaussian_kernel[5][5] = { 
	{ 0.00224214, 0.0165673, 0.0165673, 0.0165673, 0.00224214 }, 
	{ 0.0165673, 0.0450347, 0.122417, 0.0450347, 0.0165673 }, 
	{ 0.0165673, 0.122417, 0.122417, 0.122417, 0.0165673 }, 
	{ 0.0165673, 0.0450347, 0.122417, 0.0450347, 0.0165673 }, 
	{ 0.00224214, 0.0165673, 0.0165673, 0.0165673, 0.00224214} 
};

__constant int sobel_gx_kernel[3][3] = {
	{ -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, -1 }
};

__constant int sobel_gy_kernel[3][3] = {
	{ -1,-2,-1 },
	{ 0, 0, 0 },
	{ 1, 2, 1 }
};

__kernel void gaussian_blur(
	__global uchar *inImage,
	__global uchar *outImage,
	size_t rows,
	size_t cols)
{
	int sum = 0;
	size_t row = get_global_id(0);
	size_t col = get_global_id(1);
	size_t pos = row * cols + col;

	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			sum += gaussian_kernel[i][j] * inImage[(i + row + -1)*cols + (j + col + -1)];

	outImage[pos] = min(255, max(0, sum));
}