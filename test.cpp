#include <iostream>
#include <algorithm>

#include "include/CLUtility.h"
#include "include/CLEvent.h"
#include "include/CLContext.h"
#include "include/CLBuffer.h"
#include "include/CLProgram.h"

int main()
{
	// create a context for the second GPU with one command queues
	clp::Context context(CL_DEVICE_TYPE_GPU, 1, 1);

	// create and build a program
	clp::Program program(context);
	program.setSource(
	"kernel void saxpy(global float *x, global float *y, float a)\n"
	"{\n"
	"	const uint index = get_global_id(0);\n"
	"	x[index] += a*y[index];\n"
	"}\n"
	);
	program.build();
	
	// obtain a kernel object
	clp::Kernel<void(float*, float*, float)> saxpy = program.getKernel<void(float*, float*, float)>("saxpy");
	
	// create device buffers
	clp::Buffer<float> x(context, 1024);
	clp::Buffer<float> y(context, 1024);

	// map the buffers
	clp::Event xevent = x.map();
	clp::Event yevent = y.map();
	
	// fill them with data and unmap
	xevent.wait();
	std::fill(x.begin(), x.end(), 45);
	x.unmap();
	
	yevent.wait();
	std::fill(y.begin(), y.end(), 3);
	y.unmap();

	// execute kernel
	saxpy(clp::Worksize(1024,256), x, y, 13);
	
	return 0;
}
