#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
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

Mat read_unary(const char *filename) {
    ifstream file(filename);
    string line;

    vector<double> values;

    int num_of_rows = 0;

    while (getline(file, line)) {
        stringstream linestream(line);
        string cell;

        while (getline(linestream, cell, ',')) {
            values.push_back(stod(cell));
        }
        num_of_rows++;
    }

    Mat mat = Mat(num_of_rows, (int) values.size() / num_of_rows, CV_64F, &values[0]);

    return mat;
}

int main() {
    Mat I = imread("../banana3.png");
    namedWindow("original_image", CV_WINDOW_AUTOSIZE);
    imshow("original_image", I);
    waitKey(0);

    Mat pdf_background = read_unary("../pdf_b.csv");
    Mat pdf_foreground = read_unary("../pdf_f.csv");


    return 0;
}

