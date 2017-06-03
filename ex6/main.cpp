#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "maxflow-v3.04.src/graph.h"

using namespace std;
using namespace cv;

const double W_COEFF = 1e-2;

Graph<double, double, double> constructGraph(Mat fg_ucost, Mat bg_ucost, double bcost_coeff) {
    int nb_rows = fg_ucost.rows;
    int nb_cols = fg_ucost.cols;
    Graph<double, double, double> graph(nb_rows * nb_cols, 2 * nb_rows * nb_cols - nb_rows - nb_cols);
    // create nodes
    graph.add_node(nb_rows * nb_cols);
    // add unary / binary costs
    for (int i = 0; i < nb_rows; ++i) {
        for (int j = 0; j < nb_cols; ++j) {
            graph.add_tweights(i * nb_cols + j, fg_ucost.at<double>(i, j), bg_ucost.at<double>(i, j));
            if (j != nb_cols - 1) {
                graph.add_edge(i * nb_cols + j, i * nb_cols + j + 1, bcost_coeff, bcost_coeff);
            }
            if (i != nb_rows - 1) {
                graph.add_edge(i * nb_cols + j, (i + 1) * nb_cols + j, bcost_coeff, bcost_coeff);
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

Mat label_image(Graph<double, double, double> &graph, int num_of_rows, int num_of_cols) {
    Mat resultImage(num_of_rows, num_of_cols, CV_8U);

    for (int i = 0; i < num_of_rows; ++i) {
        for (int j = 0; j < num_of_cols; ++j) {
            resultImage.at<uchar>(i, j) = (uchar) ((graph.what_segment(i * num_of_cols + j) ==
                                                    Graph<double, double, double>::SOURCE) ? 0xFF
                                                                                           : 0x00);
        }
    }
    return resultImage;
}

int main() {
    Mat I = imread("../banana3.png");
    namedWindow("original_image", CV_WINDOW_AUTOSIZE);
    imshow("original_image", I);

    Mat pdf_background = read_unary("../pdf_b.csv");
    Mat pdf_foreground = read_unary("../pdf_f.csv");

    Graph<double, double, double> graph = constructGraph(pdf_foreground, pdf_background, W_COEFF);
    graph.maxflow();
    Mat resultImage = label_image(graph, pdf_background.rows, pdf_background.cols);

    namedWindow("result image", CV_WINDOW_AUTOSIZE);
    imshow("result image", resultImage);
    waitKey(0);

    return 0;
}