/*
 * msrc_dataset_unary.cpp
 *
 *  Created on: May 24, 2016
 *      Author: summer
 */

#include <stdio.h>      // input/output functions
#include <stdlib.h>     // general functions
#include <iostream>         // streaming functions
#include <string.h>
#include <vector>
#include <math.h>
#include <cmath>
#include <map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN 1
#include <winsock2.h>
#include <windows.h>
#else

#include <netinet/in.h>

#endif

#include <stdint.h>
#include <utility>
#include <vector>
#include "../maxflow/graph.h"


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

typedef Graph<double, double, double> Graph3D;
typedef pair<pair<int, int>, bool> BinaryKey;
typedef pair<int, int> Coord2;

std::pair<std::vector<float>, std::vector<float> > get_bcost_h_v(cv::Mat rgb, const float lamb) {
    cv::Size sz = rgb.size();
    int nb_rows = rgb.rows, nb_cols = rgb.cols;

    std::vector<float> hbcost(nb_rows * (nb_cols - 1)), vbcost((nb_rows - 1) * nb_cols);
    for (int i = 0; i < nb_rows; ++i) {
        for (int j = 0; j < nb_cols; ++j) {
            if (j != nb_cols - 1) {
                cv::Vec3f diff = (cv::Vec3f) rgb.at<cv::Vec3b>(i, j) - (cv::Vec3f) rgb.at<cv::Vec3b>(i, j + 1);
                hbcost[i * (nb_cols - 1) + j] = exp(-lamb * (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) / (255 * 255));
            }
            if (i != nb_rows - 1) {
                cv::Vec3f diff = (cv::Vec3f) rgb.at<cv::Vec3b>(i, j) - (cv::Vec3f) rgb.at<cv::Vec3b>(i + 1, j);
                vbcost[i * nb_cols + j] = exp(-lamb * (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) / (255 * 255));
            }
        }
    }
    return std::make_pair(hbcost, vbcost);
}

const float LAMBDA = 0.5;


class ProbImage {
public:
    ProbImage();

    ProbImage(const ProbImage &o);

    ~ProbImage();

    ProbImage &operator=(const ProbImage &o);

    // Load and save
    void load(const char *file);

    void save(const char *file);

    void decompress(const char *file);

    // Properties
    int width() const { return width_; }

    int height() const { return height_; }

    int depth() const { return depth_; }

    // Conversion operations
    void boostToProb();

    void probToBoost();

    // Data access
    // i = column, j = row, k = class id
    const float &operator()(int i, int j, int k) const { return data_[(j * width_ + i) * depth_ + k]; }

    float &operator()(int i, int j, int k) { return data_[(j * width_ + i) * depth_ + k]; }

    const float *data() const { return data_; }

    float *data() { return data_; }

protected:
    float *data_;
    int width_, height_, depth_;
};


ProbImage::ProbImage() : data_(NULL), width_(0), height_(0), depth_(0) {
}

ProbImage::ProbImage(const ProbImage &o) : width_(o.width_), height_(o.height_), depth_(o.depth_) {
    data_ = new float[width_ * height_ * depth_];
    memcpy(data_, o.data_, width_ * height_ * depth_ * sizeof(float));
}

ProbImage &ProbImage::operator=(const ProbImage &o) {
    width_ = o.width_;
    height_ = o.height_;
    depth_ = o.depth_;
    if (data_) delete[] data_;
    data_ = new float[width_ * height_ * depth_];
    memcpy(data_, o.data_, width_ * height_ * depth_ * sizeof(float));
    return *this;
}

ProbImage::~ProbImage() {
    if (data_) delete[] data_;
}

static void readBuf32(FILE *fp, unsigned int size, uint32_t *buf) {
    fread(buf, sizeof(*buf), size, fp);
    for (int i = 0; i < size; i++)
        buf[i] = ntohl(buf[i]);
}

static void writeBuf32(FILE *fp, unsigned int size, uint32_t *buf) {
    uint32_t sbuf[(1 << 13)];
    for (int i = 0; i < size; i += (1 << 13)) {
        for (int j = 0; j < (1 << 13) && i + j < size; j++)
            sbuf[j] = htonl(buf[i + j]);
        fwrite(sbuf, sizeof(*sbuf), (size - i) < (1 << 13) ? (size - i) : (1 << 13), fp);
    }
}

void ProbImage::load(const char *file) {
    FILE *fp = fopen(file, "rb");
    uint32_t buf[4];
    readBuf32(fp, 3, buf);
    width_ = buf[0];
    height_ = buf[1];
    depth_ = buf[2];
    if (data_) delete[] data_;
    data_ = new float[width_ * height_ * depth_];
    readBuf32(fp, width_ * height_ * depth_, (uint32_t *) data_);
    fclose(fp);
}

void ProbImage::save(const char *file) {
    FILE *fp = fopen(file, "wb");
    uint32_t buf[] = {width_, height_, depth_};
    writeBuf32(fp, 3, buf);
    writeBuf32(fp, width_ * height_ * depth_, (uint32_t *) data_);
    fclose(fp);
}

static int cmpKey(const void *a, const void *b) {
    const int *ia = (const int *) a, *ib = (const int *) b;
    for (int i = 0;; i++)
        if (ia[i] < ib[i])
            return -1;
        else if (ia[i] > ib[i])
            return 1;
    return 0;
}

void ProbImage::boostToProb() {
    for (int i = 0; i < width_ * height_; i++) {
        float *dp = data_ + i * depth_;
        float mx = dp[0];
        for (int j = 1; j < depth_; j++)
            if (mx < dp[j])
                mx = dp[j];
        float nm = 0;
        for (int j = 0; j < depth_; j++)
            nm += (dp[j] = exp((dp[j] - mx)));
        nm = 1.0 / nm;
        for (int j = 0; j < depth_; j++)
            dp[j] *= nm;
    }
}

void ProbImage::probToBoost() {
    for (int i = 0; i < width_ * height_; i++) {
        float *dp = data_ + i * depth_;
        for (int j = 0; j < depth_; j++)
            dp[j] = log(dp[j]);
    }
}

void ProbImage::decompress(const char *file) {
    FILE *fp = fopen(file, "rb");
    uint32_t buf[5];
    readBuf32(fp, 5, buf);
    width_ = buf[0];
    height_ = buf[1];
    depth_ = buf[2];
    int M = buf[3];
    float eps = *(float *) (buf + 4);

    uint32_t *ids = new uint32_t[width_ * height_];
    int32_t *ukeys = new int32_t[M * depth_];
    readBuf32(fp, M * depth_, (uint32_t *) ukeys);
    readBuf32(fp, width_ * height_, ids);

    if (data_) delete[] data_;
    data_ = new float[width_ * height_ * depth_];

    for (int i = 0; i < width_ * height_; i++) {
        int32_t *k = ukeys + ids[i] * depth_;
        for (int j = 0; j < depth_; j++)
            data_[i * depth_ + j] = k[j] * eps;
    }

    fclose(fp);
    delete[] ids;
    delete[] ukeys;
}

Graph3D* constructGraph(vector<Coord2>& non_alpha_node_indices,
						map<Coord2, pair<float, float> >& unary_energies,
						map<BinaryKey, pair<float, Coord2> >& binary_energies) {
	map<Coord2, int> coord2index;
	int i = 0;
	for (auto it = non_alpha_node_indices.begin(); it < non_alpha_node_indices.end(); it++, i++){
		coord2index.insert(make_pair(*it, i));
	}
	Graph3D *graph = new Graph3D(non_alpha_node_indices.size(), binary_energies.size());
    // create nodes
	graph->add_node(non_alpha_node_indices.size());
    // add unary / binary costs
	i = 0;
	for (auto it = non_alpha_node_indices.begin(); it < non_alpha_node_indices.end(); it++, i++){
		pair<float, float> t_weights = unary_energies[*it];
		// add unary weights
		graph->add_tweights(i, t_weights.second, t_weights.first);
		// add binary weights if exists
		if (binary_energies.find(make_pair(*it, false)) != binary_energies.end()){
			pair<float, Coord2> bval = binary_energies[make_pair(*it, false)];
			graph->add_edge(i, coord2index[bval.second], bval.first, bval.first);
		}
		if (binary_energies.find(make_pair(*it, true)) != binary_energies.end()){
			pair<float, Coord2> bval = binary_energies[make_pair(*it, true)];
			graph->add_edge(i, coord2index[bval.second], bval.first, bval.first);
		}

	}
    return graph;
}

void get_label_from_graph(Mat& label, Graph3D *graph, vector<Coord2>& non_alpha_node_indices, uchar alpha){
	graph->maxflow();
	int i = 0;
	for (auto it = non_alpha_node_indices.begin(); it < non_alpha_node_indices.end(); it++, i++){
			if (graph->what_segment(i) == Graph<double, double, double>::SOURCE){
				label.at<uchar>(it->first, it->second) = alpha;
//                cout << it->first << " " << it->second << endl;
			}
	}
	delete graph;
}

void set_unaries(Mat &unary, vector<float> &binary_weights,
                 uchar k, map<pair<int, int>, pair<float, float> > &unary_energies, int i_idx, int j_idx,
                 int weight_idx) {

    uchar label = unary.at<uchar>(j_idx, i_idx);
    if (label != k) {
        pair<int, int> index_pair = make_pair(j_idx, i_idx);
        float weight = binary_weights[weight_idx];

        if (unary_energies.find(index_pair) != unary_energies.end()) {
            pair<float, float> energies = make_pair(0, weight);
            unary_energies.insert(make_pair(index_pair, energies));

        } else {
            pair<float, float> current_energy = unary_energies[index_pair];
            current_energy.second += weight;
            unary_energies[index_pair] = current_energy;
        }
    }
}


void calculate_energies(const ProbImage &prob, Mat &unary, Mat &rgb) {
    auto binary_weights = get_bcost_h_v(rgb, LAMBDA);

    for (uchar k = 0; k < prob.depth(); ++k) {

		vector<Coord2> non_alpha_node_indices;
		map<Coord2, pair<float, float> > unary_energies;
		map<BinaryKey, pair<float, Coord2> > binary_energies;

        for (int i = 0; i < unary.cols; ++i) {
            for (int j = 0; j < unary.rows; ++j) {
                if (unary.at<uchar>(j, i) != k) {
                    auto index_pair = make_pair(j, i);
                    non_alpha_node_indices.push_back(index_pair);

                    if (unary_energies.find(index_pair) != unary_energies.end()) {
                        pair<float, float> energies = make_pair(-log(prob(i, j, k)), -log(prob(i, j, unary.at<uchar>(j, i))));
                        unary_energies.insert(make_pair(index_pair, energies));
                    } else {
                        pair<float, float> current_energy = unary_energies[index_pair];
                        current_energy.first += -log(prob(i, j, k));
                        current_energy.second += -log(prob(i, j, unary.at<uchar>(j, i)));
                        unary_energies[index_pair] = current_energy;
                    }

                    if (j != unary.rows - 1) {
                        if (unary.at<uchar>(j+1, i) != k) {
                            if (unary.at<uchar>(j+1, i) != unary.at<uchar>(j, i)) {
                                binary_energies.insert(make_pair(make_pair(make_pair(j, i), true),
                                                                 make_pair(binary_weights.second[j * unary.cols + i],
                                                                           make_pair(j + 1, i))));
                            } else {
                                binary_energies.insert(make_pair(make_pair(make_pair(j, i), true),
                                                                 make_pair(0, make_pair(j + 1, i))));
                            }
                        }
                    }
                    if (i != unary.cols - 1) {
                        if (unary.at<uchar>(j, i+1) != k) {
                            if (unary.at<uchar>(j, i+1) != unary.at<uchar>(j, i)) {
                                binary_energies.insert(make_pair(make_pair(make_pair(j, i), false),
                                                                 make_pair(binary_weights.first[j * (unary.cols - 1) + i],
                                                                           make_pair(j, i + 1))));
                            } else {
                                binary_energies.insert(make_pair(make_pair(make_pair(j, i), false),
                                                                 make_pair(0, make_pair(j, i + 1))));
                            }

                        }
                    }

                } else {
                    if (j != unary.rows - 1) {
                        set_unaries(unary, binary_weights.second, k, unary_energies, i, j + 1, j * unary.cols + i);
                    }
                    if (i != unary.cols - 1) {
                        set_unaries(unary, binary_weights.first, k, unary_energies, i + 1, j, j * unary.cols + i);
                    }
                }
            }
        }
		// do graph stuff
		Graph3D *graph = constructGraph(non_alpha_node_indices, unary_energies, binary_energies);
		get_label_from_graph(unary, graph, non_alpha_node_indices, k);
    }
}

bool not_same(Mat a, Mat b){
	return countNonZero(a != b) != 0;
}

int main(int argc, char **argv) {
    printf("load unary potentials\n");

    // the filenames of image and its unary should match!
    std::string rgbfile = "../1_9_s.bmp", unaryfile = "../1_9_s.c_unary";

    // load the probability distribution data
    ProbImage prob;
    prob.decompress(unaryfile.c_str());
    printf("decompressed the probability distribution, %d, %d, %d\n", prob.width(), prob.height(), prob.depth());

    // demo, generate a segmentation where the label corresponds to the highest class probability
	// contrust the CRF and solve the segmentation with your implementation of alpha-expansion
    cv::Mat rgb = cv::imread(rgbfile, CV_LOAD_IMAGE_COLOR);
    cv::Mat unary(rgb.size(), CV_8U, cv::Scalar(0));

//    prob.probToBoost();

    for (int i = 0; i < unary.cols; ++i) {
        for (int j = 0; j < unary.rows; ++j) {
            float maxprob = 0.f;
            int lid = 0;
            for (int k = 0; k < prob.depth(); ++k) {
                if (maxprob < prob(i, j, k)) {
                    maxprob = prob(i, j, k);
                    lid = k;
                }
            }

            unary.at<uchar>(j, i) = static_cast<uchar>(lid);
        }
    }

    cv::imshow("unary before", 10 * unary);

    cv::Mat old_unary;
	int counter = 1;
	do{
		cout << "iteration " << counter++ << endl;
		unary.copyTo(old_unary);
		calculate_energies(prob, unary, rgb);
	} while (not_same(unary, old_unary));

	// visualization
    cv::imshow("rgb", rgb);
    cv::imshow("unary", 10 * unary);
    cv::waitKey();
    return 0;
}