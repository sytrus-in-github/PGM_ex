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
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN 1
#include <winsock2.h>
#include <windows.h>
#else
#include <netinet/in.h>
#endif
#include <stdint.h>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ProbImage{
public:
        ProbImage();
        ProbImage( const ProbImage & o );
        ~ProbImage();
        ProbImage & operator=( const ProbImage & o );

        // Load and save
        void load( const char * file );
        void save( const char * file );
        void decompress( const char * file );

        // Properties
        int width() const { return width_; }
        int height() const { return height_; }
        int depth() const { return depth_; }

        // Conversion operations
        void boostToProb();
        void probToBoost();

        // Data access
        // i = column, j = row, k = class id
        const float & operator()( int i, int j, int k ) const { return data_[(j*width_+i)*depth_+k]; }
        float & operator()( int i, int j, int k ) { return data_[(j*width_+i)*depth_+k]; }
        const float * data() const { return data_; }
        float * data() { return data_; }

protected:
        float * data_;
        int width_, height_, depth_;
};


ProbImage::ProbImage() :data_(NULL),width_(0),height_(0),depth_(0){
}
ProbImage::ProbImage(const ProbImage& o) : width_( o.width_ ), height_( o.height_ ), depth_( o.depth_ ){
        data_ = new float[ width_*height_*depth_ ];
        memcpy( data_, o.data_, width_*height_*depth_*sizeof(float) );
}
ProbImage& ProbImage::operator=(const ProbImage& o) {
        width_ = o.width_; height_ = o.height_; depth_ = o.depth_;
        if( data_ ) delete[] data_;
        data_ = new float[ width_*height_*depth_ ];
        memcpy( data_, o.data_, width_*height_*depth_*sizeof(float) );
		return *this;
}
ProbImage::~ProbImage(){
        if( data_ ) delete[] data_;
}
static void readBuf32( FILE * fp, unsigned int size, uint32_t * buf ){
        fread( buf, sizeof(*buf), size, fp );
        for( int i=0; i<size; i++ )
                buf[i] = ntohl( buf[i] );
}
static void writeBuf32( FILE * fp, unsigned int size, uint32_t * buf ){
        uint32_t sbuf[(1<<13)];
        for( int i=0; i<size; i+=(1<<13) ){
                for( int j=0; j<(1<<13) && i+j<size; j++ )
                        sbuf[j] = htonl( buf[i+j] );
                fwrite( sbuf, sizeof(*sbuf), (size-i) < (1<<13) ? (size-i) : (1<<13), fp );
        }
}
void ProbImage::load(const char* file) {
        FILE* fp = fopen( file, "rb" );
        uint32_t buf[4];
        readBuf32( fp, 3, buf );
        width_ = buf[0]; height_ = buf[1]; depth_ = buf[2];
        if( data_ ) delete[] data_;
        data_ = new float[ width_*height_*depth_ ];
        readBuf32( fp, width_*height_*depth_, (uint32_t*)data_ );
        fclose( fp );
}
void ProbImage::save(const char* file) {
        FILE* fp = fopen( file, "wb" );
        uint32_t buf[] = {width_, height_, depth_};
        writeBuf32( fp, 3, buf );
        writeBuf32( fp, width_*height_*depth_, (uint32_t*)data_ );
        fclose( fp );
}

static int cmpKey( const void * a, const void * b ){
        const int * ia = (const int*)a, *ib = (const int*)b;
        for( int i=0;; i++ )
                if (ia[i] < ib[i])
                        return -1;
                else if (ia[i] > ib[i])
                        return 1;
        return 0;
}
void ProbImage::boostToProb() {
        for( int i=0; i<width_*height_; i++ ){
                float * dp = data_ + i*depth_;
                float mx = dp[0];
                for( int j=1; j<depth_; j++ )
                        if (mx < dp[j])
                                mx = dp[j];
                float nm = 0;
                for( int j=0; j<depth_; j++ )
                        nm += (dp[j] = exp( (dp[j]-mx) ) );
                nm = 1.0 / nm;
                for( int j=0; j<depth_; j++ )
                        dp[j] *= nm;
        }
}
void ProbImage::probToBoost() {
        for( int i=0; i<width_*height_; i++ ){
                float * dp = data_ + i*depth_;
                for( int j=0; j<depth_; j++ )
                        dp[j] = log( dp[j] );
        }
}

void ProbImage::decompress(const char* file) {
        FILE* fp = fopen( file, "rb" );
        uint32_t buf[5];
        readBuf32( fp, 5, buf );
        width_ = buf[0]; height_ = buf[1]; depth_ = buf[2];
        int M = buf[3];
        float eps = *(float*)(buf+4);

        uint32_t * ids = new uint32_t[width_*height_];
        int32_t * ukeys = new int32_t[M*depth_];
        readBuf32( fp, M*depth_, (uint32_t*)ukeys );
        readBuf32( fp, width_*height_, ids );

        if( data_ ) delete[] data_;
        data_ = new float[ width_*height_*depth_ ];

        for( int i=0; i<width_*height_; i++ ){
                int32_t * k = ukeys + ids[i]*depth_;
                for( int j=0; j<depth_; j++ )
                        data_[ i*depth_ + j ] = k[j]*eps;
        }

        fclose( fp );
        delete [] ids;
        delete [] ukeys;
}

int
main (int argc, char** argv)
{
    printf ("load unary potentials\n");

    // the filenames of image and its unary should match!
    std::string rgbfile = "../1_27_s.bmp", unaryfile = "../1_27_s.c_unary";

    // load the probability distribution data
    ProbImage prob;
    prob.decompress(unaryfile.c_str());
    printf("decompressed the probability distribution, %d, %d, %d\n", prob.width(), prob.height(), prob.depth());

    // demo, generate a segmentation where the label corresponds to the highest class probability
    // todo, contrust the CRF and solve the segmentation with your implementation of alpha-expansion
    cv::Mat rgb = cv::imread(rgbfile, CV_LOAD_IMAGE_COLOR);
    cv::Mat unary (rgb.size(), CV_8U, cv::Scalar(0));
    for(int i = 0; i < unary.cols; ++i)
    {
        for (int j = 0; j < unary.rows; ++j)
        {
            float maxprob = 0.f;
            int lid = 0;
            for (int k = 0; k < prob.depth(); ++k)
            {
                if (maxprob < prob(i,j, k))
                {
                    maxprob  = prob(i,j,k);
                    lid = k;
                }
            }

            // times 10 for better visualization
            unary.at<uchar>(j, i) = static_cast<uchar>(10 * lid);
        }
    }

    cv::imshow("rgb", rgb);
    cv::imshow("unary", unary);
    cv::waitKey();
    return 0;
}
