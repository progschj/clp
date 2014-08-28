#ifndef CL_KERNEL_H
#define CL_KERNEL_H

#include "CLEvent.h"
#include "CLContext.h"
#include "CLImage.h"

namespace clp {
	
template<class T>
struct translate {
	typedef const T type;
};

template<class T>
struct translate<T*> {
	typedef Buffer<T> type;
};

template<class T>
class Local {
public:
	Local(size_t s) : size(s) { }
	size_t size;
};

template<class T>
struct Args {
	static void set(cl_kernel kernel, int n, T arg)
	{
		checkError(clSetKernelArg(kernel, n, sizeof(T), &arg));
	}
};
	
template<class T>
struct Args< Buffer<T> > {
	static void set(cl_kernel kernel, int n, Buffer<T> &arg)
	{
		checkError(clSetKernelArg(kernel, n, sizeof(cl_mem), arg.getMem()));
	}
};

template<class T>
struct Args< const Buffer<T> > {
	static void set(cl_kernel kernel, int n, const Buffer<T> &arg)
	{
		checkError(clSetKernelArg(kernel, n, sizeof(cl_mem), arg.getMem()));
	}
};

template<class T>
struct Args< Image2D<T> > {
	static void set(cl_kernel kernel, int n, Image2D<T> &arg)
	{
		checkError(clSetKernelArg(kernel, n, sizeof(cl_mem), arg.getMem()));
	}
};

template<class T>
struct Args< const Image2D<T> > {
	static void set(cl_kernel kernel, int n, const Image2D<T> &arg)
	{
		checkError(clSetKernelArg(kernel, n, sizeof(cl_mem), arg.getMem()));
	}
};

template<class T>
struct Args< Local<T> > {
	static void set(cl_kernel kernel, int n, Local<T> arg)
	{
		checkError(clSetKernelArg(kernel, n, sizeof(T)*arg.size, 0));
	}
};

template<class T>
void setKernelArg(cl_kernel kernel, int n, T &arg)
{
	Args<T>::set(kernel, n, arg);
}

struct Worksize {
	Worksize(size_t g1, size_t l1) : dim(1)
	{
		global[0] = g1;
		local[0] = l1;
	}
	Worksize(size_t g1, size_t g2, size_t l1, size_t l2) : dim(2)
	{
		global[0] = g1; global[1] = g2;
		local[0] = l1; local[1] = l2;
	}
	Worksize(size_t g1, size_t g2, size_t g3, size_t l1, size_t l2, size_t l3) : dim(3)
	{
		global[0] = g1; global[1] = g2; global[2] = g3;
		local[0] = l1; local[1] = l2; local[2] = l3;
	}
	size_t global[3];
	size_t local[3];
	cl_uint dim;
};

template<class F>
class Kernel {
};

template<class T0>
class Kernel<void(T0)> {
public:
	Kernel(const Context &c, cl_kernel k) : kernel(k), context(c) {}
	Kernel(const Kernel &k) : kernel(k.kernel), context(k.context)
	{
		checkError(clRetainKernel(kernel));
	}
	
	typedef typename translate<T0>::type A0;
	
	Event operator()(const Worksize &ws, A0 &arg0)
	{
		return operator()(ws, arg0, 0, 0);
	}

	Event operator()(const Worksize &ws, A0 &arg0, const Event &event)
	{
		return operator()(ws, arg0, 1, event.getEventPtr());
	}
	
	Event operator()(const Worksize &ws, A0 &arg0, cl_uint event_count, const cl_event *events)
	{
		setKernelArg(kernel, 0, arg0);
		cl_event event;
		cl_int error = clEnqueueNDRangeKernel(context.getQueue(), kernel, ws.dim, 0, ws.global, ws.local, event_count, events, &event);
		checkError(error);
		return Event(event);
	}
	
	~Kernel()
	{
		checkError(clReleaseKernel(kernel));
	}
private:
	cl_kernel kernel;
	Context context;
};

template<class T0, class T1>
class Kernel<void(T0, T1)> {
public:
	Kernel(const Context &c, cl_kernel k) : kernel(k), context(c) {}
	Kernel(const Kernel &k) : kernel(k.kernel), context(k.context)
	{
		checkError(clRetainKernel(kernel));
	}
	
	typedef typename translate<T0>::type A0;
	typedef typename translate<T1>::type A1;
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1)
	{
		return operator()(ws, arg0, arg1, 0, 0);
	}
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1, const Event &event)
	{
		return operator()(ws, arg0, arg1, 1, event.getEventPtr());
	}
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1, cl_uint event_count, const cl_event *events)
	{
		setKernelArg(kernel, 0, arg0);
		setKernelArg(kernel, 1, arg1);
		cl_event event;
		cl_int error = clEnqueueNDRangeKernel(context.getQueue(), kernel, ws.dim, 0, ws.global, ws.local, event_count, events, &event);
		checkError(error);
		return Event(event);
	}
	
	~Kernel()
	{
		checkError(clReleaseKernel(kernel));
	}
private:
	cl_kernel kernel;
	Context context;
};

template<class T0, class T1, class T2>
class Kernel<void(T0, T1, T2)> {
public:
	Kernel(const Context &c, cl_kernel k) : kernel(k), context(c) {}
	Kernel(const Kernel &k) : kernel(k.kernel), context(k.context)
	{
		checkError(clRetainKernel(kernel));
	}
	
	typedef typename translate<T0>::type A0;
	typedef typename translate<T1>::type A1;
	typedef typename translate<T2>::type A2;
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1, A2 &arg2)
	{
		return operator()(ws, arg0, arg1, arg2, 0, 0);
	}
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1, A2 &arg2, const Event &event)
	{
		return operator()(ws, arg0, arg1, arg2, 1, event.getEventPtr());
	}
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1, A2 &arg2, cl_uint event_count, const cl_event *events)
	{
		setKernelArg(kernel, 0, arg0);
		setKernelArg(kernel, 1, arg1);
		setKernelArg(kernel, 2, arg2);
		cl_event event;
		cl_int error = clEnqueueNDRangeKernel(context.getQueue(), kernel, ws.dim, 0, ws.global, ws.local, event_count, events, &event);
		checkError(error);
		return Event(event);
	}
	
	~Kernel()
	{
		checkError(clReleaseKernel(kernel));
	}
private:
	cl_kernel kernel;
	Context context;
};


template<class T0, class T1, class T2, class T3>
class Kernel<void(T0, T1, T2, T3)> {
public:
	Kernel(const Context &c, cl_kernel k) : kernel(k), context(c) {}
	Kernel(const Kernel &k) : kernel(k.kernel), context(k.context)
	{
		checkError(clRetainKernel(kernel));
	}
	
	typedef typename translate<T0>::type A0;
	typedef typename translate<T1>::type A1;
	typedef typename translate<T2>::type A2;
	typedef typename translate<T3>::type A3;
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1, A2 &arg2, A3 &arg3)
	{
		return operator()(ws, arg0, arg1, arg2, arg3, 0, 0);
	}
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1, A2 &arg2, A3 &arg3, const Event &event)
	{
		return operator()(ws, arg0, arg1, arg2, arg3, 1, event.getEventPtr());
	}
	
	Event operator()(const Worksize &ws, A0 &arg0, A1 &arg1, A2 &arg2, A3 &arg3, cl_uint event_count, const cl_event *events)
	{
		setKernelArg(kernel, 0, arg0);
		setKernelArg(kernel, 1, arg1);
		setKernelArg(kernel, 2, arg2);
		setKernelArg(kernel, 3, arg3);
		cl_event event;
		cl_int error = clEnqueueNDRangeKernel(context.getQueue(), kernel, ws.dim, 0, ws.global, ws.local, event_count, events, &event);
		checkError(error);
		return Event(event);
	}
	
	~Kernel()
	{
		checkError(clReleaseKernel(kernel));
	}
private:
	cl_kernel kernel;
	Context context;
};
}

#endif
