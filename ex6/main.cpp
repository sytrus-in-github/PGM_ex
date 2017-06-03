#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "maxflow/graph.h"

using namespace std;
using namespace cv;

const double W_COEFF = 7e-6;
typedef Graph<double, double, double> Graph3D;

Graph3D* constructGraph(Mat& fg_ucost, Mat& bg_ucost, const double bcost_coeff) {
    int nb_rows = fg_ucost.rows;
    int nb_cols = fg_ucost.cols;
	Graph3D *graph = new Graph3D(nb_rows * nb_cols, 2 * nb_rows * nb_cols - nb_rows - nb_cols);
    // create nodes
    graph -> add_node(nb_rows * nb_cols);
    // add unary / binary costs
    for (int i = 0; i < nb_rows; ++i) {
        for (int j = 0; j < nb_cols; ++j) {
            graph -> add_tweights(i * nb_cols + j, fg_ucost.at<double>(i, j), bg_ucost.at<double>(i, j));
            if (j != nb_cols - 1) {
                graph -> add_edge(i * nb_cols + j, i * nb_cols + j + 1, bcost_coeff, bcost_coeff);
            }
            if (i != nb_rows - 1) {
                graph -> add_edge(i * nb_cols + j, (i + 1) * nb_cols + j, bcost_coeff, bcost_coeff);
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

	int num_of_cols = (int)values.size() / num_of_rows;
	// cout << num_of_rows << ' ' << num_of_cols<< ' ' << num_of_cols * num_of_rows << ' ' << values.size() << endl;
	Mat mat = Mat(num_of_rows, num_of_cols, CV_64FC1);
	memcpy(mat.data, values.data(), values.size()*sizeof(double));
	return mat;
}

Mat label_image(Graph3D *graph, int num_of_rows, int num_of_cols) {
    Mat resultImage(num_of_rows, num_of_cols, CV_8U);

    for (int i = 0; i < num_of_rows; ++i) {
        for (int j = 0; j < num_of_cols; ++j) {
            resultImage.at<uchar>(i, j) = (uchar) ((graph->what_segment(i * num_of_cols + j) ==
                                                    Graph<double, double, double>::SOURCE) ? 0xFF
                                                                                           : 0x00);
        }
    }
    return resultImage;
}

int testGraph()
{
	typedef Graph<int, int, int> GraphType;
	GraphType *g = new GraphType(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);

	g->add_node();
	g->add_node();

	g->add_tweights(0,   /* capacities */  1, 5);
	g->add_tweights(1,   /* capacities */  2, 6);
	g->add_edge(0, 1,    /* capacities */  3, 4);

	int flow = g->maxflow();

	printf("Flow = %d\n", flow);
	printf("Minimum cut:\n");
	if (g->what_segment(0) == GraphType::SOURCE)
		printf("node0 is in the SOURCE set\n");
	else
		printf("node0 is in the SINK set\n");
	if (g->what_segment(1) == GraphType::SOURCE)
		printf("node1 is in the SOURCE set\n");
	else
		printf("node1 is in the SINK set\n");

	delete g;

	return 0;
}

int main() {
	// testGraph();

	cout << "reading unary costs\n";

    Mat pdf_background = read_unary("../pdf_b.csv");
    Mat pdf_foreground = read_unary("../pdf_f.csv");

	// cout << pdf_background.size() << endl;
	// cout << pdf_background.rows << ' ' << pdf_background.cols << endl;
	// cout << pdf_background.at<double>(479, 639) << " " << pdf_foreground.at<double>(0, 0) << endl;
	
	cout << "building graph\n";

	Graph3D *graph = constructGraph(pdf_foreground, pdf_background, W_COEFF);
    
	cout << "computing maxflow\n";

	graph->maxflow();
    Mat resultImage = label_image(graph, pdf_background.rows, pdf_background.cols);

	cout << "showing results\n";

	Mat I = imread("../banana3.png");
	namedWindow("original_image", CV_WINDOW_AUTOSIZE);
	imshow("original_image", I);
	Mat Igmm = imread("../GMM_segmented_banana.png");
	namedWindow("GMM segmentation", CV_WINDOW_AUTOSIZE);
	imshow("GMM segmentation", Igmm);
    namedWindow("graphcut segmentation", CV_WINDOW_AUTOSIZE);
	imshow("graphcut segmentation", resultImage);
    waitKey(0);

    return 0;
}