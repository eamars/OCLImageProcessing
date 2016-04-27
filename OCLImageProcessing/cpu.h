#pragma once

void CPUConvolve(const unsigned char * in, unsigned char * out, const int dataSizeX, const int dataSizeY, const float *kernel, int kernelSizeX, int kernelSizeY);
void CPUConvolveFast(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY,
	float* kernel, int kernelSizeX, int kernelSizeY);