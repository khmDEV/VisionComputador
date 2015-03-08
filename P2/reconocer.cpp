/*
 * Autores: Aron Collados (626558)
 *          Cristian Roman (646564)
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include "fileSystem.h"
using namespace cv;
using namespace std;

/*
 * Grises
 */
Mat Grises(Mat bgrMap){
  Mat dst;
  cvtColor( bgrMap, dst, CV_RGB2GRAY );
  return dst;
}

/*
 * Otsu effect
 */
Mat Otsu(Mat src){
  Mat dst;
  threshold(src, dst, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  return dst;
}

/*
 * Adatative effect
 */
Mat adaptative(Mat src){
   Mat dst;
   adaptiveThreshold(src, dst,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,75,10);
   return dst;
}

/*
 * Main principal
 */
int main(int argc, char *argv[]) {
    if(argc<3){
    	cerr << argv[0] << " nomfich nomobj [MINSIZE]" << endl;
        return -1;
    }
    std::string image = argv[1],obj = argv[2],size=argc>=4?argv[3]:"1000";
    int MINSIZE=atoi(size.c_str());

    Mat bgrMap = imread(image, CV_LOAD_IMAGE_COLOR); //Carga la imagen recibida por parametro
    if (bgrMap.empty()) {
 	std::cerr << "Could not open file " << image << std::endl;
        return -1;
    }

    Mat binary=(Otsu(Grises(bgrMap)));
    /*
     * Get Contors
     */
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( binary, contours, hierarchy,CV_RETR_LIST, CV_CHAIN_APPROX_TC89_L1, Point(0, 0) );
   
    /*
     * Calculate Moments
     */
    vector<Moments> mu;

    for( int i = 0; i < contours.size(); i++ ){ 
	Moments m=moments( contours[i], false );
	if(MINSIZE<m.m00){	
		mu.push_back(m); 
	}
    }
    if(mu.size()<1){
    	cerr << "Error: Object not found with minimum size: " << MINSIZE << endl;
	return -1;
    }else if(mu.size()>1){
    	cerr << "Error: Has found " << mu.size() << " objects whith minimum size: " << MINSIZE << endl;
	return -1;
    }
    /*
     * Learn
     */
    addMoment(obj.c_str(),mu.at(0));
    vector<vector<float> > vec=getMoments(obj.c_str());
    for (int i = 0; i < vec.size(); i++) {
    	vector<float> vf=vec.at(i);
    	for (int o = 0; o < vf.size(); o++) {
    		cout << vf.at(o)<<"\n"<<endl;
	}
    }
}
