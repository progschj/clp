#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <cmath>

#include <CL/cl.h>

#include "include/CLUtility.h"
#include "include/CLEvent.h"
#include "include/CLContext.h"
#include "include/CLBuffer.h"
#include "include/CLProgram.h"


int main()
{
	clp::Context context(CL_DEVICE_TYPE_GPU, 1, 2);


	clp::Program program(context);
	program.setSource(
	"float2 cmul(float2 a, float2 b)\n"
	"{\n"
	"	return (float2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);"
	"}\n" 
	
	"__kernel void saxpy(__global float *x, __global float *y, float a)\n"
	"{\n"
	"	const uint index = get_global_id(0);\n"
	"	x[index] += a*y[index];\n"
	"}\n"
	);
	program.build();
	clp::Kernel<void(float*, float*, float)> saxpy = program.getKernel<void(float*, float*, float)>("saxpy");
	
	clp::Buffer<float> x(context, 1024);
	clp::Buffer<float> y(context, 1024);

	context.setCurrentQueue(0);
	x.map();
	y.map().wait();
	
	
	std::fill(x.begin(), x.end(), 45);
	std::fill(y.begin(), y.end(), 3);

	x.unmap();
	y.unmap();

	saxpy(clp::Worksize(1024,256), x, y, 13);

	x.map().wait();
	for(size_t i = 0;i<x.size();++i)
	{
		std::cout << x[i] << std::endl;
	}
	x.unmap();
	
	return 0;
}
