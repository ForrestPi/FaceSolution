
//https://www.jianshu.com/p/5dc844002d72

#include <opencv2/opencv.hpp>
#include"ncnn_mtcnn_tld_so.hpp"
#include <stdio.h>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>

#include"ndarray_converter.h"

using namespace cv;
using namespace std;

namespace py = pybind11;

class FaceTracker :private faceTrack
{
public:
	FaceTracker() { faceTrack(); };
	~FaceTracker() {};

public:
	void trackerInit(const std::string& model_path, const int min_face) {
		this->Init(model_path, min_face);
	}
	std::vector<int> trackerUpdate(cv::Mat& image) {
		cv::Rect rect;
		this->DetectFace(rect, image);
		return vector<int>{rect.x, rect.y, rect.x + rect.width, rect.y + rect.height};
	};

public:
	std::string version = "v1.0.0";

};



#if 0

int main() {
	cv::VideoCapture capture;
	capture.open("./test.avi");

	cv::Mat frame;
	faceTrack tracker;
	std::string modelPath = "./models";
	int minFace = 40;
	tracker.Init(modelPath, minFace);

	while (capture.read(frame)) {
		int q = cv::waitKey(1);
		if (q == 27) break;
		cv::Rect result;
		double t1 = (double)getTickCount();
		tracker.DetectFace(result, frame);
		printf("total %gms\n", ((double)getTickCount() - t1) * 1000 / getTickFrequency());
		printf("------------------\n");
		rectangle(frame, result, Scalar(0, 0, 255), 2);
		imshow("frame", frame);
		//      outputVideo << frame;
	}
	//  outputVideo.release();
	capture.release();
	cv::destroyAllWindows();
	return 0;
}


#endif // 0


#if 1
PYBIND11_MODULE(face_tracking_demo, m) {

	NDArrayConverter::init_numpy();

	py::class_<FaceTracker>(m, "FaceTracker")
		.def(py::init<>())
		.def("trackerInit", &FaceTracker::trackerInit, py::arg("model_path"), py::arg("min_face"))
		.def("trackerUpdate", &FaceTracker::trackerUpdate, py::arg("img"));
}

#endif
