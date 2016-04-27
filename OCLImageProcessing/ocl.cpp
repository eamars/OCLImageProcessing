#include "ocl.h"

#include <CL/cl.hpp>
#include <string>
#include <iostream>
#include <vector>

#include "utils.h"

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::cout;

#define FREE(ptr, free_val) \
    if (ptr != free_val)    \
    {                       \
        free(ptr);          \
        ptr = free_val;     \
    }


/*
Detect platform, create context, device list and command queue
*/
void CLInit(oclHandle *handle, const string kernelFile, vector<string> kernelNames)
{
	// find available platforms
	std::vector<cl::Platform> allPlatforms;
	cl::Platform targetPlatform;

	cl::Platform::get(&allPlatforms);

	if (!(allPlatforms.size() > 0))
		throw (string("CLInit()::Error: No platforms found (cl::Platform::get())"));

	// select target platform, select the first by default
	int sel = 0;
	targetPlatform = allPlatforms[sel];
	std::cout << "\n";
	for (int idx = 0; idx < allPlatforms.size(); idx++)
	{
		std::cout << "[";
		std::cout << (idx == sel ? "*]" : " ]");
		std::cout << allPlatforms[idx].getInfo<CL_PLATFORM_VENDOR>() << std::endl;

	}

	// create OCL context
	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)targetPlatform(), 0 };
	handle->context = cl::Context(CL_DEVICE_TYPE_CPU, cprops);

	// detect OCL devices
	handle->devices = handle->context.getInfo<CL_CONTEXT_DEVICES>();

	// create OCL command queue
	handle->queue = cl::CommandQueue(handle->context, handle->devices[0]);

	// load cl scripts
	string sourceStr = FileToString(kernelFile);
	cl::Program::Sources sources(1, std::make_pair(sourceStr.c_str(), sourceStr.length()));
	handle->program = cl::Program(handle->context, sources);

	// build for capacible devices
	handle->program.build(handle->devices);

	// create kernel
	// handle->kernel = cl::Kernel(handle->program, kernelName.c_str());
	for (int i = 0; i < kernelNames.size(); i++)
	{
		cl::Kernel kernel = cl::Kernel(handle->program, kernelNames[i].c_str());
		handle->kernels.push_back(kernel);
	}
}

void CLSetKernelArgs(oclHandle *handle, oclBuffer *oclBuffers, int idx)
{
	// input image
	handle->kernels[idx].setArg(0, oclBuffers->inputCL);

	// output image
	handle->kernels[idx].setArg(1, oclBuffers->outputCL);

	// data size x
	handle->kernels[idx].setArg(2, oclBuffers->dataSizeX);

	// data size y
	handle->kernels[idx].setArg(3, oclBuffers->dataSizeY);

	// kernel
	handle->kernels[idx].setArg(4, oclBuffers->kernelCL);

	// kernel size x
	handle->kernels[idx].setArg(5, oclBuffers->kernelSizeX);

	// kernel size y
	handle->kernels[idx].setArg(6, oclBuffers->kernelSizeY);
}

void CLEnqueueKernel(oclHandle *handle, oclBuffer *oclBuffers, int idx)
{
	cl::Event event;
	handle->queue.enqueueNDRangeKernel(
		handle->kernels[idx],
		cl::NDRange(),
		cl::NDRange(oclBuffers->dataSizeX, oclBuffers->dataSizeY),
		cl::NDRange(32, 32),
		0,
		&event
	);

	event.wait();
}


void CLBufferInit(oclHandle *handle, oclBuffer * oclBuffers, hostBuffer * hostBuffers)
{
	oclBuffers->dataSizeX = hostBuffers->dataSizeX;
	oclBuffers->dataSizeY = hostBuffers->dataSizeY;
	oclBuffers->kernelSizeX = hostBuffers->kernelSizeX;
	oclBuffers->kernelSizeY = hostBuffers->kernelSizeY;

	oclBuffers->inputCL = cl::Buffer(
		handle->context,
		CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(unsigned char) * hostBuffers->dataSizeX * hostBuffers->dataSizeY,
		hostBuffers->in
	);

	oclBuffers->outputCL = cl::Buffer(
		CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(unsigned char) * hostBuffers->dataSizeX * hostBuffers->dataSizeY,
		hostBuffers->out
	);

	oclBuffers->kernelCL = cl::Buffer(
		CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(float) * hostBuffers->kernelSizeX * hostBuffers->kernelSizeY,
		hostBuffers->kernel
	);

}
