#ifndef CL_EVENT_H
#define CL_EVENT_H

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>

#include "CLUtility.h"

namespace clp
{

class Event {
public:
	Event() : assigned(false) { }
	Event(cl_event e) : assigned(true), event(e) { }
	Event(const Event &e) : assigned(e.assigned), event(e.event) 
	{ 
		if(assigned)
			clRetainEvent(event);
	}
	Event& operator=(const Event &e)
	{
		if(assigned)
			clReleaseEvent(event);
		assigned = e.assigned;
		event = e.event;
		if(assigned)
			clRetainEvent(event);
		return *this;
	}
	
	cl_int getStatus()
	{
		cl_int status;
		size_t size;
		checkError(clGetEventInfo (event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, &size));
		return status;
	}
	
	void wait()
	{
		checkError(clWaitForEvents(1,&event));
	}
	
	operator cl_event() const { return event; }
	
	cl_event const* getEventPtr() const { return &event; }
	
	~Event()
	{
		if(assigned)
			clReleaseEvent(event);
	}
private:
	bool assigned;	
	cl_event event;
};

}

#endif
