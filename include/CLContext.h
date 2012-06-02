#ifndef CL_CONTEXT_H
#define CL_CONTEXT_H

#include <stdexcept>
#include <string>
#include <memory>

#include <CL/cl.h>

#include "CLUtility.h"

namespace clp
{

class Context {
public:
	Context(cl_device_type type = CL_DEVICE_TYPE_ALL, cl_uint requested_device = 0, cl_uint queuecount = 1)
		: data(new ContextData)
	{
		cl_uint platforms, devices;
		cl_int error;
		error = clGetPlatformIDs(1, &(data->platform), &platforms);
		checkError(error);
		
		error = clGetDeviceIDs(data->platform, type, 0, 0, &devices);
		checkError(error);
			
		std::unique_ptr<cl_device_id[]> device_ids(new cl_device_id[devices]);
			
		error = clGetDeviceIDs(data->platform, type, devices, device_ids.get(), 0);
		checkError(error);
		
		if(requested_device>=devices)
			throw std::runtime_error("no such device");
			
		data->device = device_ids[requested_device];
			
		cl_context_properties properties[]={
			CL_CONTEXT_PLATFORM, (cl_context_properties)(data->platform),
		0};
				
		data->context = clCreateContext(properties, 1, &(data->device), 0, 0, &error);
		checkError(error);
		data->queues.resize(queuecount);
		for(size_t i = 0;i<queuecount;++i)
		{ 
			data->queues[i] = clCreateCommandQueue(data->context, data->device, 0, &error);
			checkError(error);
		}
		data->current_queue = 0;
	}
	
	Context(const Context &c)
		: data(c.data)
	{
	}
	
	cl_platform_id getPlatform() const { return data->platform; }
	cl_device_id getDevice() const { return data->device; }
	cl_context getContext() const { return data->context; }
	cl_command_queue getQueue() const { return data->queues[data->current_queue]; }
	size_t getQueueCount() const { return data->queues.size(); }
	void setCurrentQueue(size_t i) const { data->current_queue = i; }
	size_t getCurrentQueue() const { return data->current_queue; }
private:
	struct ContextData {
		cl_platform_id platform;
		cl_device_id device;
		cl_context context;
		std::vector<cl_command_queue> queues;
		size_t current_queue;
		~ContextData()
		{
			cl_int error;
			error = clReleaseContext(context);
			checkError(error);
			for(size_t i = 0;i<queues.size();++i)
			{
				error = clReleaseCommandQueue(queues[i]);
				checkError(error);
			}
		}
	};
	
	std::shared_ptr<ContextData> data;
};


}

#endif
