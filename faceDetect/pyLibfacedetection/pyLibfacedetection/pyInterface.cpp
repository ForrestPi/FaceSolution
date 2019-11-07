#include<array>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include<opencv2/opencv.hpp>
#include<facedetectcnn.h>
#include"ndarray_converter.h"

namespace py = pybind11;


class Face {

public:
	std::array<int, 4> rect;  //[xmin,ymin,xmax,ymax]
	int angle;
	int neighbors;

public:
	Face() {};
	Face(std::array<int, 4>& rect, int angle, int neighbors) {
		this->rect = rect;
		this->angle = angle;
		this->neighbors = neighbors;
	}


	~Face() {};
};

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000


std::vector<Face> facedetect(cv::Mat& image) {
	int * pResults = NULL;
	//pBuffer is used in the detection functions.
	//If you call functions in multiple threads, please create one buffer for each thread!
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		std::runtime_error("Can not alloc buffer.\n");
		//fprintf(stderr, "Can not alloc buffer.\n");

	}


	///////////////////////////////////////////
	// CNN face detection 
	// Best detection rate
	//////////////////////////////////////////
	//!!! The input image must be a RGB one (three-channel)
	//!!! DO NOT RELEASE pResults !!!
	pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);

	//printf("%d faces detected.\n", (pResults ? *pResults : 0));
	cv::Mat result_cnn = image.clone();;
	//print the detection results

	std::vector<Face> faces;
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];

		std::array<int, 4> arr;
		arr[0] = x;
		arr[1] = y;
		arr[2] = x + w;
		arr[3] = y + h;
		faces.push_back(Face(arr, angle, neighbors));

		//printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
		//rectangle(result_cnn, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
	}

	return faces;


}


//std::vector<Face> facedetect(std::string filename) {
//
//  cv::Mat image = cv::imread(filename);
//  if (image.empty())
//  {
//      std::runtime_error("image read failed!\n");
//  }
//
//  int * pResults = NULL;
//  //pBuffer is used in the detection functions.
//  //If you call functions in multiple threads, please create one buffer for each thread!
//  unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
//  if (!pBuffer)
//  {
//      std::runtime_error("Can not alloc buffer.\n");
//      //fprintf(stderr, "Can not alloc buffer.\n");
//
//  }
//
//
//  ///////////////////////////////////////////
//  // CNN face detection 
//  // Best detection rate
//  //////////////////////////////////////////
//  //!!! The input image must be a RGB one (three-channel)
//  //!!! DO NOT RELEASE pResults !!!
//  pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);
//
//  //printf("%d faces detected.\n", (pResults ? *pResults : 0));
//  cv::Mat result_cnn = image.clone();;
//  //print the detection results
//
//  std::vector<Face> faces;
//  for (int i = 0; i < (pResults ? *pResults : 0); i++)
//  {
//      short * p = ((short*)(pResults + 1)) + 142 * i;
//      int x = p[0];
//      int y = p[1];
//      int w = p[2];
//      int h = p[3];
//      int neighbors = p[4];
//      int angle = p[5];
//
//      std::array<int, 4> arr;
//      arr[0] = x;
//      arr[1] = y;
//      arr[2] = x + w;
//      arr[3] = y + h;
//      faces.push_back(Face(arr, angle, neighbors));
//
//      //printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
//      //rectangle(result_cnn, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
//  }
//
//  return faces;
//
//
//}
//

PYBIND11_MODULE(pyLibfacedetection_cnn, m) {

	m.doc() = "Simple python warper of libfacedetection-cnn";


	NDArrayConverter::init_numpy();

	py::class_<Face>(m, "Face")
		.def(py::init())
		.def_readwrite("rect", &Face::rect)
		.def_readwrite("angle", &Face::angle)
		.def_readwrite("neighbors", &Face::neighbors);

	m.def("facedetect", &facedetect);




}


