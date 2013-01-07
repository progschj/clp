#ifndef CL_BUFFER_H
#define CL_BUFFER_H

#include "CLEvent.h"
#include "CLContext.h"

namespace clp {

template<class T>
class Buffer {
public:
	typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef const ptrdiff_t difference_type;
    typedef size_t size_type;

	Buffer(const Context &c, size_t s)
		: host_ptr(0), buffersize(s), context(c)
	{
		cl_int error;
		buffer = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, buffersize*sizeof(value_type), 0, &error);
		checkError(error);
	}

	Event map(cl_map_flags flags = CL_MAP_READ | CL_MAP_WRITE)
	{
		return map(flags, 0, 0);
	}

	Event map(cl_map_flags flags, const Event &event)
	{
		return map(flags, 1, event.getEventPtr());
	}
	
	Event map(cl_map_flags flags, cl_uint event_count, const cl_event *events)
	{
		check_unmapped();
		cl_int error;
		cl_event e;
		host_ptr = static_cast<value_type*>(clEnqueueMapBuffer(context.getQueue(), buffer, CL_FALSE, flags, 0, buffersize*sizeof(value_type), event_count, events, &e, &error));
		checkError(error);
		event = Event(e);
		return event;
	}
	
	Event unmap()
	{
		return unmap(0, 0);
	}
	
	Event unmap(const Event &event)
	{
		return unmap(1, event.getEventPtr());
	}
	
	Event unmap(cl_uint event_count, const cl_event *events)
	{
		check_mapped();
		cl_event e;
		cl_int error = clEnqueueUnmapMemObject(context.getQueue(), buffer, host_ptr, event_count, events, &e);
		host_ptr = 0;
		checkError(error);
		event = Event(e);
		return event;
	}

	Event read(value_type *destination)
	{
		return read(destination, 0, 0);
	}
	
	Event read(value_type *destination, const Event &event)
	{
		return read(destination, 1, event.getEventPtr());
	}

	Event read(value_type *destination, cl_uint event_count, const cl_event *events)
	{
		check_unmapped();
		cl_event e;
		cl_int error = clEnqueueReadBuffer (context.getQueue(), buffer, CL_FALSE, 0, buffersize*sizeof(value_type), destination, event_count, events, &e);
		checkError(error);
		event = Event(e);
		return event;
	}

	Event readRange(size_t offset, size_t length, value_type *destination)
	{
		return readRange(offset, length, destination, 0, 0);
	}
	
	Event readRange(size_t offset, size_t length, value_type *destination, const Event &event)
	{
		return readRange(offset, length, destination, 1, event.getEventPtr());
	}
	
	Event readRange(size_t offset, size_t length, value_type *destination, cl_uint event_count, const cl_event *events)
	{
		if(offset+length>buffersize)
			throw std::runtime_error("buffer too short");
		check_unmapped();
		cl_event e;
		cl_int error = clEnqueueReadBuffer (context.getQueue(), buffer, CL_FALSE, offset*sizeof(value_type), length*sizeof(value_type), destination, event_count, events, &e);
		checkError(error);
		event = Event(e);
		return event;
	}
	
	Event write(const value_type *source)
	{
		return write(source, 0, 0);
	}
	
	Event write(const value_type *source, const Event &event)
	{
		return write(source, 1, event.getEventPtr());
	}
	
	Event write(const value_type *source, cl_uint event_count, const cl_event *events)
	{
		check_unmapped();
		cl_event e;
		cl_int error = clEnqueueWriteBuffer (context.getQueue(), buffer, CL_FALSE, 0, buffersize*sizeof(value_type), source, event_count, events, &e);
		checkError(error);
		event = Event(e);
		return event;
	}

	Event writeRange(size_t offset, size_t length, const value_type *source)
	{
		return writeRange(offset, length, source, 0, 0);
	}
	
	Event writeRange(size_t offset, size_t length, const value_type *source, const Event &event)
	{
		return writeRange(offset, length, source, 1, event.getEventPtr());
	}
	
	Event writeRange(size_t offset, size_t length, const value_type *source, cl_uint event_count, const cl_event *events)
	{
		if(offset+length>buffersize)
			throw std::runtime_error("buffer too short");
		check_unmapped();
		cl_event e;
		cl_int error = clEnqueueWriteBuffer (context.getQueue(), buffer, CL_FALSE, offset*sizeof(value_type), length*sizeof(value_type), source, event_count, events, &e);
		checkError(error);
		event = Event(e);
		return event;
	}
    
    inline reference operator[](size_t i)
    {
        check_mapped();
        return host_ptr[i];
    }

    inline const_reference operator[](size_t i) const
    {
        check_mapped();
        return host_ptr[i];
    }
    
    inline value_type* data() { check_mapped(); return host_ptr; }
    inline const value_type* data() const { check_mapped(); return host_ptr; }
    inline iterator begin() { check_mapped(); return host_ptr; }
    inline const_iterator begin() const { check_mapped(); return host_ptr; }
    inline iterator end() { check_mapped(); return host_ptr+buffersize; }
    inline const_iterator end() const { check_mapped(); return host_ptr+buffersize; }
    
    inline size_type size() const { return buffersize; }

	const cl_mem* getMem() const { return &buffer; }
	Event getLastEvent() { return event; }
	
	bool isMapped() const { return host_ptr != 0; }
		
	~Buffer()
	{
		clReleaseMemObject(buffer);
	}
private:
	Buffer(const Buffer&) { }
	Buffer& operator=(const Buffer&) { return *this; }
	
	inline void check_mapped() const
    {
        if(!host_ptr)
            throw std::runtime_error("Buffer not mapped");
    }

    inline void check_unmapped() const
    {
        if(host_ptr)
            throw std::runtime_error("Buffer mapped");
    }
    
	value_type *host_ptr;
	size_t buffersize;
	cl_mem buffer;
	Event event;
	Context context;
};

}

#endif
