#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include "maxflow-v3.04.src/graph.h"

using namespace std;
using namespace cv;
int main(){ 
	Mat I = imread("../banana3.png");
	namedWindow("original_image", CV_WINDOW_AUTOSIZE);
	imshow("original_image", I);
	waitKey(0);
	return 0;
}