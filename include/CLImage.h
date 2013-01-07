#ifndef CL_IMAGE_H
#define CL_IMAGE_H

#include "CLEvent.h"
#include "CLContext.h"

namespace clp {

template<class T>
class Image2D {
public:
	typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef const ptrdiff_t difference_type;
    typedef size_t size_type;

	Image2D(const Context &c, size_t width, size_t height)
		: host_ptr(0), width_(width), height_(height), context(c)
	{
        cl_image_format format;
        format.image_channel_data_type = type2format<T>::type;
		format.image_channel_order = type2format<T>::order;
		cl_int error;
		buffer = clCreateImage2D(context.getContext(), CL_MEM_READ_WRITE, &format, width_, height_, 0, 0, &error);
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
        const size_t origin[] = {0, 0, 0};
        const size_t region[] = {width_, height_, 1};
		host_ptr = static_cast<value_type*>(clEnqueueMapImage(context.getQueue(), buffer, CL_FALSE, flags, origin, region, &image_row_pitch, &image_slice_pitch, event_count, events, &e, &error));
		checkError(error);
        image_row_pitch /= sizeof(value_type);
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
    
    inline reference operator()(size_t i, size_t j)
    {
        check_mapped();
        return host_ptr[j*image_row_pitch + i];
    }

    inline const_reference operator()(size_t i, size_t j) const
    {
        check_mapped();
        return host_ptr[j*image_row_pitch + i];
    }
    
    inline value_type* data() { check_mapped(); return host_ptr; }
    inline const value_type* data() const { check_mapped(); return host_ptr; }
    inline size_type row_pitch() const { return image_row_pitch; }
    
    inline size_type width() const { return width_; }
    inline size_type height() const { return height_; }

	const cl_mem* getMem() const { return &buffer; }
	Event getLastEvent() { return event; }
	
	bool isMapped() const { return host_ptr != 0; }
		
	~Image2D()
	{
		clReleaseMemObject(buffer);
	}
private:
	Image2D(const Image2D&) { }
	Image2D& operator=(const Image2D&) { return *this; }
	
	inline void check_mapped() const
    {
        if(!host_ptr)
            throw std::runtime_error("Image not mapped");
    }

    inline void check_unmapped() const
    {
        if(host_ptr)
            throw std::runtime_error("Image mapped");
    }
    
	value_type *host_ptr;
	size_t width_, height_;
    size_t image_row_pitch, image_slice_pitch;
	cl_mem buffer;
	Event event;
	Context context;
};



template<class T>
class Image3D {
public:
	typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef const ptrdiff_t difference_type;
    typedef size_t size_type;

	Image3D(const Context &c, size_t width, size_t height, size_t depth)
		: host_ptr(0), width_(width), height_(height), depth_(depth), context(c)
	{
        cl_image_format format;
        format.image_channel_data_type = type2format<T>::type;
		format.image_channel_order = type2format<T>::order;
		cl_int error;
		buffer = clCreateImage3D(context.getContext(), CL_MEM_READ_WRITE, &format, width_, height_, depth_, 0, 0, 0, &error);
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
        const size_t origin[] = {0, 0, 0};
        const size_t region[] = {width_, height_, depth_};
		host_ptr = static_cast<value_type*>(clEnqueueMapImage(context.getQueue(), buffer, CL_FALSE, flags, origin, region, &image_row_pitch, &image_slice_pitch, event_count, events, &e, &error));
		checkError(error);
        image_row_pitch /= sizeof(value_type);
		image_slice_pitch /= sizeof(value_type);
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
    
    inline reference operator()(size_t i, size_t j, size_t k)
    {
        check_mapped();
        return host_ptr[k*image_slice_pitch + j*image_row_pitch + i];
    }

    inline const_reference operator()(size_t i, size_t j, size_t k) const
    {
        check_mapped();
        return host_ptr[k*image_slice_pitch + j*image_row_pitch + i];
    }
    
    inline value_type* data() { check_mapped(); return host_ptr; }
    inline const value_type* data() const { check_mapped(); return host_ptr; }
    inline size_type row_pitch() const { return image_row_pitch; }
    inline size_type slice_pitch() const { return image_slice_pitch; }
    
    inline size_type width() const { return width_; }
    inline size_type height() const { return height_; }
    inline size_type depth() const { return depth_; }

	const cl_mem* getMem() const { return &buffer; }
	Event getLastEvent() { return event; }
	
	bool isMapped() const { return host_ptr != 0; }
		
	~Image3D()
	{
		clReleaseMemObject(buffer);
	}
private:
	Image3D(const Image3D&) { }
	Image3D& operator=(const Image3D&) { return *this; }
	
	inline void check_mapped() const
    {
        if(!host_ptr)
            throw std::runtime_error("Image not mapped");
    }

    inline void check_unmapped() const
    {
        if(host_ptr)
            throw std::runtime_error("Image mapped");
    }
    
	value_type *host_ptr;
	size_t width_, height_, depth_;
    size_t image_row_pitch, image_slice_pitch;
	cl_mem buffer;
	Event event;
	Context context;
};

}

#endif
