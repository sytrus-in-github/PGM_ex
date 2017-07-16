#include <stdio.h>      
#include <stdlib.h>     
#include <iostream>     
#include <string.h>
#include <vector>
#include <math.h>
#include <cmath>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int
main(int argc, char** argv)
{
    // change to your file directory
    std::string rgbfile = "/home/summer/PhD/teaching/pgm-ss16/exercisesheets/program-repo/images/1_29_s.bmp";
    if (argc == 2)
        rgbfile = argv[1];

    std::string unaryfile = rgbfile;
    unaryfile.replace(unaryfile.end()-4, unaryfile.end(), ".yml");

    // load image and its unary
    cv::Mat rgb = cv::imread(rgbfile, CV_LOAD_IMAGE_COLOR);
    cv::Mat1f unary;
    cv::FileStorage store (unaryfile, cv::FileStorage::READ);
    store["unary"] >> unary;
    store.release ();

    cv::imshow("rgb", rgb);
    cv::imshow("unary", unary);
    cv::waitKey();

    return 0;
}
