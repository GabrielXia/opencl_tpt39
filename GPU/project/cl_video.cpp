#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"

#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024

using namespace cv;
using namespace std;
#define SHOW

Mat getGaussianKernel2d(int rows, int cols, double sigmax, double sigmay )
    {
        Mat kernel = Mat::zeros(rows, cols, CV_32FC1); 

        float meanj = (kernel.rows-1)/2, 
              meani = (kernel.cols-1)/2,
              sum = 0,
              temp= 0;

        int sigma=2*sigmay*sigmax;
        for(int j=0;j<kernel.rows;j++)
            for(int i=0;i<kernel.cols;i++)
            {
                temp = exp( -((j-meanj)*(j-meanj) + (i-meani)*(i-meani))  / (sigma));
                kernel.at<float>(j,i) = temp;
				sum += kernel.at<float>(j,i);
            }

        if(sum != 0)
            return kernel /= sum;
        else return Mat();
    }

void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}
unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  return output;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}



void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

void paddingTransform(Mat img, float* mat2){
	Mat padding;
	copyMakeBorder(img, padding, 1, 1, 1, 1, BORDER_REPLICATE);
	Mat temp(padding.size(), CV_32FC1);
	padding.convertTo(temp, CV_32FC1);
	float* array = (float *) malloc(sizeof(float)*(padding.rows)*(padding.cols));
	array = (float* )temp.data;
	int blur_size = 3;
	int row_s = blur_size * blur_size;
	int wide = (int) img.cols;
	int height = (int) img.rows;
	int wide_mat1 = wide + 2;
	for (int i=0; i< wide*height*row_s; i++){
		int pixel = i / row_s;
		int order = i % row_s;
		int x_start = pixel % wide;
		int y_start = pixel / wide;
		int x_1 = x_start + order% blur_size;
		int y_1 = y_start + order/ blur_size;
		mat2[i] = array[x_1 + y_1 * wide_mat1];
	}
}

void compute(cl_context context, cl_command_queue queue, cl_kernel kernel, float* in1, float* in2, float *output, int wide, int height) {
	cl_mem input_a_buf; // num_devices elements
	cl_mem input_b_buf; // num_devices elements
	cl_mem output_buf; // num_devices elements
	int status;
	int row_s = 9;
	input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
			wide*height*row_s* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        	row_s* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        	wide*height* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
	
	cl_event write_event[2];
    cl_event kernel_event,finish_event;
    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
        	0, wide*height*row_s* sizeof(float), in1, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
        	0, row_s* sizeof(float), in2, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &row_s);
    checkError(status, "Failed to set argument 4");

	const size_t global_work_size = wide * height;
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        	&global_work_size, NULL, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");
	
	status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
    	    0, wide*height* sizeof(float), output, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to set");
	clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
	clReleaseMemObject(input_a_buf);
	clReleaseMemObject(input_b_buf);
	clReleaseMemObject(output_buf);
}
int main(int, char**)
{	
	// opencl setup
	char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;
	//-------------------------------finish setup


    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./cl.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);
	
    VideoWriter outputVideo;                                        // Open the output
        outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	double diff,tot;
	int count=0;
	const char *windowName = "filter";   // Name shown in the GUI window.
    #ifdef SHOW
    namedWindow(windowName); // Resizable window, might not work on Windows.
    #endif

	clGetPlatformIDs(1, &platform, NULL);
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
        printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
        printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
        printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

        context_properties[1] = (cl_context_properties)platform;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device, 0, NULL);

        unsigned char **opencl_program=read_file("cl_video.cl");
        program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
        if (program == NULL)
        {
             printf("Program creation failed\n");
            return 1;
        }
        int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
        kernel = clCreateKernel(program, "matrixMul", NULL);

    while (true) {
		time_t start,end;
        Mat cameraFrame,displayframe;
		count=count+1;
		if(count > 100) break;
        camera >> cameraFrame;
		time (&start);
        Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
        Mat grayframe,edge_x,edge_y,edge;
    	cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
		int blur_size = 3;
	    int row_s = blur_size * blur_size;
    	int wide = (int) grayframe.cols;
    	int height = (int) grayframe.rows;
		
		// gausian filter
		float *mat2 = (float *) malloc(sizeof(float)*row_s*wide*height);
		paddingTransform(grayframe, mat2);
		Mat filter_g_mat = getGaussianKernel2d(blur_size, blur_size, 1, 1);
		float *filter_g=(float*)filter_g_mat.data;
		float *output=(float *) malloc(sizeof(float)*wide*height);
		compute(context, queue, kernel, mat2, filter_g, output, wide, height);

		// scharr filter 1
		float filter_scharr[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};	
		Mat img(cameraFrame.size(), CV_32FC1, output);
		paddingTransform(img, mat2);
		compute(context, queue, kernel, mat2, filter_scharr, output, wide, height); 
	
		// scharr filter 2
        float filter_scharr2[9] = {-1, -2, 1, 0, 0, 0, 1, 2, 1};
        Mat img2(cameraFrame.size(), CV_32FC1, output);
        paddingTransform(img2, mat2);
        compute(context, queue, kernel, mat2, filter_scharr2, output, wide, height);

		time (&end);
        diff = difftime (end,start);
        printf ("GPU took %.8lf seconds to run.\n", diff );
	
		//Mat result(cameraFrame.size(), CV_8UC1, output);
		unsigned char* a = (unsigned char *) malloc(sizeof(unsigned char)*wide*height); 
		for (int i=0; i<wide*height;i++) {
			a[i] = (unsigned char)output[i];			
		}
		Mat result(cameraFrame.size(), CV_8UC1, a);
	
        cvtColor(result, displayframe, CV_GRAY2BGR);
		outputVideo << displayframe;
	#ifdef SHOW
        imshow(windowName, displayframe);
	#endif
		diff = difftime (end,start);
		tot+=diff;
	}

	clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
	clReleaseProgram(program);
    clReleaseContext(context);
    clFinish(queue);

	outputVideo.release();
	camera.release();
  	//printf ("FPS %.2lf .\n", 299.0/tot );

    return EXIT_SUCCESS;

}
