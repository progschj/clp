#ifndef CL_PROGRAM_H
#define CL_PROGRAM_H

#include "CLKernel.h"

namespace clp
{

class Program {
public:
	Program(const Context &c)
		: context(c)
	{		
	}

	void setSource(const std::string &s)
	{
		source = s;
	}

	void build()
	{
		const char *s[] = {source.c_str()};
		size_t length = source.size();
		cl_int error;
		program = clCreateProgramWithSource(context.getContext(), 1, s, &length, &error);
		checkError(error);
		
		error = clBuildProgram(program, 0, 0, "", 0, 0);
		if(error != CL_SUCCESS)
		{
			size_t length;
			clGetProgramBuildInfo(program, context.getDevice(), CL_PROGRAM_BUILD_LOG, 0, 0, &length);
			std::string log; log.resize(length);
			clGetProgramBuildInfo(program, context.getDevice(), CL_PROGRAM_BUILD_LOG, length, &log[0], 0);
			throw std::runtime_error(log);
		}
	}
	
	template<class T>
	Kernel<T> getKernel(const std::string &name)
	{
		cl_kernel kernel;
		cl_int error;
		kernel = clCreateKernel(program, name.c_str(), &error);
		checkError(error);
		return Kernel<T>(context, kernel);
	}
	
private:
	cl_program program;
	std::string source;
	Context context;
};

} // end namespace clp


#endif
