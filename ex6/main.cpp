#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include "maxflow-v3.04.src/graph.h"
#include "opencv/ml.h"

using namespace std;
using namespace cv;

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

