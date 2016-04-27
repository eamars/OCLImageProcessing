#pragma once

#include <CL/cl.hpp>
#include <string>
#include <vector>

typedef struct oclBufferStruct
{
	cl::Buffer inputCL;
	cl::Buffer kernelCL;
	cl::Buffer outputCL;

	int dataSizeX;
	int dataSizeY;
	int kernelSizeX;
	int kernelSizeY;
} oclBuffer;

typedef struct hostBufferStruct
{
	unsigned char * in;
	unsigned char *out;
	int dataSizeX;
	int dataSizeY;

	float *kernel;
	int kernelSizeX;
	int kernelSizeY;
	
} hostBuffer;

typedef struct oclHandleStruct
{
	cl::Context					context;
	std::vector<cl::Device>		devices;
	cl::CommandQueue			queue;
	cl::Program					program;
	// cl::Kernel					kernel;
	std::vector<cl::Kernel>		kernels;
	
} oclHandle;


void CLInit(oclHandle *handle, const std::string kernelFile, std::vector<std::string> kernelNames);
void CLSetKernelArgs(oclHandle *handle, oclBuffer *oclBuffers, int idx);
void CLEnqueueKernel(oclHandle *handle, oclBuffer *oclBuffers, int idx);
void CLBufferInit(oclHandle *handle, oclBuffer *oclBuffers, hostBuffer *hostBuffers);
