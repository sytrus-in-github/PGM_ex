#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv/cv.h"
#include "opencv/ml.h" 
#include<iostream>
#include "maxflow-v3.04.src/graph.h"

using namespace std;
using namespace cv;

const double W_COEFF = 1e-2;

Graph<double, double, double> constructGraph(Mat fg_ucost, Mat bg_ucost, double bcost_coeff){
	int nb_rows = fg_ucost.rows;
	int nb_cols = fg_ucost.cols;
	Graph<double, double, double> graph(nb_rows * nb_cols, 2 * nb_rows * nb_cols - nb_rows - nb_cols);
	// create nodes 
	graph.add_node(nb_rows * nb_cols);
	// add unary / binary costs
	for (int i = 0; i < nb_rows; ++i){
		for (int j = 0; j < nb_cols; ++j){
			graph.add_tweights(i*nb_cols + j, fg_ucost.at<double>(i, j), bg_ucost.at<double>(i, j));
			if (j != nb_cols - 1){
				graph.add_edge(i*nb_cols + j, i*nb_cols + j + 1, bcost_coeff, bcost_coeff);
			}
			if (i != nb_rows - 1){
				graph.add_edge(i*nb_cols + j, (i + 1)*nb_cols + j, bcost_coeff, bcost_coeff);
			}
		}
	}
	return graph;
}

int main(){
	Mat I = imread("../banana3.png");
	namedWindow("original_image", CV_WINDOW_AUTOSIZE);
	imshow("original_image", I);
	waitKey(0);
	return 0;
}