#ifdef _DEBUG
#include <iostream>
#endif

#include "gpu.h"
#include "ocl.h"

using std::string;

void GPUConvolve(oclHandle *handle, oclBuffer *oclBuffers, int idx)
{
	CLSetKernelArgs(handle, oclBuffers, idx);
	CLEnqueueKernel(handle, oclBuffers, idx);
}