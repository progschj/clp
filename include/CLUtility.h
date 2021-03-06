#ifndef CLP_UTILITY_H 
#define CLP_UTILITY_H

#include <stdexcept>
#include <string>
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace clp
{

#define OPENCL_ERROR_CASE(ERROR) case ERROR: return #ERROR ;

inline std::string getStringFromError(cl_int error)
{
	switch(error)
	{		
		OPENCL_ERROR_CASE(CL_SUCCESS)
		OPENCL_ERROR_CASE(CL_DEVICE_NOT_FOUND)
		OPENCL_ERROR_CASE(CL_DEVICE_NOT_AVAILABLE)
		OPENCL_ERROR_CASE(CL_COMPILER_NOT_AVAILABLE)
		OPENCL_ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE)
		OPENCL_ERROR_CASE(CL_OUT_OF_RESOURCES)
		OPENCL_ERROR_CASE(CL_OUT_OF_HOST_MEMORY)
		OPENCL_ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE)
		OPENCL_ERROR_CASE(CL_MEM_COPY_OVERLAP)
		OPENCL_ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH)
		OPENCL_ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED)
		OPENCL_ERROR_CASE(CL_BUILD_PROGRAM_FAILURE)
		OPENCL_ERROR_CASE(CL_MAP_FAILURE)
		OPENCL_ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET)
		OPENCL_ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
		OPENCL_ERROR_CASE(CL_INVALID_VALUE)
		OPENCL_ERROR_CASE(CL_INVALID_DEVICE_TYPE)
		OPENCL_ERROR_CASE(CL_INVALID_PLATFORM)
		OPENCL_ERROR_CASE(CL_INVALID_DEVICE)
		OPENCL_ERROR_CASE(CL_INVALID_CONTEXT)
		OPENCL_ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES)
		OPENCL_ERROR_CASE(CL_INVALID_COMMAND_QUEUE)
		OPENCL_ERROR_CASE(CL_INVALID_HOST_PTR)
		OPENCL_ERROR_CASE(CL_INVALID_MEM_OBJECT)
		OPENCL_ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
		OPENCL_ERROR_CASE(CL_INVALID_IMAGE_SIZE)
		OPENCL_ERROR_CASE(CL_INVALID_SAMPLER)
		OPENCL_ERROR_CASE(CL_INVALID_BINARY)
		OPENCL_ERROR_CASE(CL_INVALID_BUILD_OPTIONS)
		OPENCL_ERROR_CASE(CL_INVALID_PROGRAM)
		OPENCL_ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE)
		OPENCL_ERROR_CASE(CL_INVALID_KERNEL_NAME)
		OPENCL_ERROR_CASE(CL_INVALID_KERNEL_DEFINITION)
		OPENCL_ERROR_CASE(CL_INVALID_KERNEL)
		OPENCL_ERROR_CASE(CL_INVALID_ARG_INDEX)
		OPENCL_ERROR_CASE(CL_INVALID_ARG_VALUE)
		OPENCL_ERROR_CASE(CL_INVALID_ARG_SIZE)
		OPENCL_ERROR_CASE(CL_INVALID_KERNEL_ARGS)
		OPENCL_ERROR_CASE(CL_INVALID_WORK_DIMENSION)
		OPENCL_ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE)
		OPENCL_ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE)
		OPENCL_ERROR_CASE(CL_INVALID_GLOBAL_OFFSET)
		OPENCL_ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST)
		OPENCL_ERROR_CASE(CL_INVALID_EVENT)
		OPENCL_ERROR_CASE(CL_INVALID_OPERATION)
		OPENCL_ERROR_CASE(CL_INVALID_GL_OBJECT)
		OPENCL_ERROR_CASE(CL_INVALID_BUFFER_SIZE)
		OPENCL_ERROR_CASE(CL_INVALID_MIP_LEVEL)
		OPENCL_ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE)
		OPENCL_ERROR_CASE(CL_INVALID_PROPERTY)
		default: return "unknown error code";
	}
}

#undef OPENCL_ERROR_CASE

inline void checkError(cl_int error)
{
	if(error != CL_SUCCESS)
		throw std::runtime_error(getStringFromError(error));
}

template<class T>
struct type2format {
};

#define OPENCL_TYPE2DEFINE(T,VALUE)                                 \
template<>                                                          \
struct type2format<T> {                                             \
    static const cl_channel_type type = VALUE;                      \
    static const cl_channel_order order = CL_R;                     \
};                                                                  \
template<>                                                          \
struct type2format<T##2> {                                          \
    static const cl_channel_type type = VALUE;                      \
    static const cl_channel_order order = CL_RG;                    \
};                                                                  \
template<>                                                          \
struct type2format<T##4> {                                          \
    static const cl_channel_type type = VALUE;                      \
    static const cl_channel_order order = CL_RGBA;                  \
};                                                                  \


OPENCL_TYPE2DEFINE(cl_char, CL_SIGNED_INT8)
OPENCL_TYPE2DEFINE(cl_short, CL_SIGNED_INT16)
OPENCL_TYPE2DEFINE(cl_int, CL_SIGNED_INT32)
OPENCL_TYPE2DEFINE(cl_uchar, CL_UNSIGNED_INT8)
OPENCL_TYPE2DEFINE(cl_ushort, CL_UNSIGNED_INT16)
OPENCL_TYPE2DEFINE(cl_uint, CL_UNSIGNED_INT32)
OPENCL_TYPE2DEFINE(cl_float, CL_FLOAT)

#undef OPENCL_TYPE2DEFINE

}

#endif
