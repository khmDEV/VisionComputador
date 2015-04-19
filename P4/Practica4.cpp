/*
 * Autores: Aron Collados (626558)
 *          Cristian Roman (646564)
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

int MIN_MATCHES=10;
int WINDOWS_TYPE=CV_WINDOW_NORMAL;
Mat mount(Mat original,Mat next){
    // detecting keypoints
    SurfFeatureDetector detector(400);
    std::vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(original, keypoints1);
    detector.detect(next, keypoints2);

    // computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(original, keypoints1, descriptors1);
    extractor.compute(next, keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    // drawing the results
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(original, keypoints1, next, keypoints2, matches, img_matches);
    imshow("matches", img_matches);
     imwrite("_M.png", img_matches);
    if((int)matches.size()>=MIN_MATCHES){
    	int a=matches.at(0).imgIdx;
	int b=matches.at(0).trainIdx;
	int min=matches.at(0).distance;
	for(int i=0;i<matches.size();i++){
	    if(min>matches.at(i).distance){
		a=matches.at(i).imgIdx;
		b=matches.at(i).trainIdx;
		min=matches.at(i).distance;
            }
	}
	int xA=keypoints1.at(a).pt.x;
	int yA=keypoints1.at(a).pt.y;
	int xB=keypoints2.at(b).pt.x;
	int yB=keypoints2.at(b).pt.y;

//circle(original,Point(xA,yA),1,Scalar(255,0,0),20);
//circle(next,Point(xB,yB),1,Scalar(255,0,0),20);

	int topMargin=yB-yA;topMargin=topMargin<0?0:topMargin;
	int bottonMargin=(next.rows-yB)-(original.rows-yA);bottonMargin=bottonMargin<0?0:bottonMargin;
	int leftMargin=xB-xA;leftMargin=leftMargin<0?0:leftMargin;
	int rigthMargin=(next.cols-xB)-(original.cols-xA);rigthMargin=rigthMargin<0?0:rigthMargin;
	

	Mat nueva(original.rows+topMargin+bottonMargin,original.cols+leftMargin+rigthMargin, original.type(), Scalar(255,0,0));

	Mat subImg(nueva, Rect(leftMargin>0?0:rigthMargin, topMargin>0?0:bottonMargin, next.cols, next.rows));

	//nueva.colRange(leftMargin>=0?1:next.cols-rigthMargin, nueva.cols-1).rowRange(topMargin>=0?1:next.rows-bottonMargin, nueva.rows-1);
	//nueva(Range(topMargin>=0?1:next.rows-bottonMargin, nueva.rows-1),Range(leftMargin>=0?1:next.cols-rigthMargin, nueva.cols-1));
	next.copyTo(subImg);	

	Mat subImg2(nueva, Rect(leftMargin, topMargin, original.cols, original.rows));

	//nueva.colRange(leftMargin==0?1:leftMargin, nueva.cols-1).rowRange(topMargin==0?1:topMargin, nueva.rows-1);
	//nueva(Range(topMargin==0?1:topMargin, nueva.rows-1),Range(leftMargin==0?1:leftMargin, nueva.cols-1));
	original.copyTo(subImg2);	
    	
	imshow("Nueva", nueva);
	return nueva;
    }
    return original;
}


/*
 * Main principal
 */
int main(int argc, char *argv[]) {
    char key=0;
    string image;
    int i=1;
    VideoCapture TheVideoCapturer;
    Mat bgrMap,captura,nueva,anterior;
    namedWindow("Original", WINDOWS_TYPE);
    namedWindow("Comparacion", WINDOWS_TYPE);
    namedWindow("matches", WINDOWS_TYPE);
    namedWindow("Nueva", WINDOWS_TYPE);
    while (key != 27){
    	if(!TheVideoCapturer.isOpened()){
   		if (argc <= i) {
        		cout << "Introduza la ruta de la imagen;" << endl; 
        		cin>> image;
    		} else {
        		image = argv[i];
    		}
	}

    	captura = imread(image, CV_LOAD_IMAGE_COLOR); //Carga la imagen recibida por parametro

    	if (captura.empty()&&!TheVideoCapturer.isOpened()) 
        {
		TheVideoCapturer.open(atoi(image.c_str())); //Abre la videocamara
	}
   	if (!TheVideoCapturer.isOpened()&&captura.empty()) {
 		std::cerr << "Could not open file " << image << std::endl;
        	return -1;
    	}
    	if (!captura.empty()){
    		bgrMap=captura;
    	}else{
    		TheVideoCapturer >> bgrMap;
    	}


    	
	if(!anterior.empty()){
		nueva=mount(bgrMap,anterior);
    		imshow("Comparacion", anterior);
	}else{
		nueva=bgrMap;
	}
    	imshow("Original", bgrMap);
	if(captura.empty()){ //Se esta usando una camara
            TheVideoCapturer >> bgrMap;
    	} 

	cout<<"Pulsa una tecla para continuar"<<endl;
	key=waitKey(0);

    	if(key==32){
     		imwrite("_F.png", bgrMap);
		if(!anterior.empty()){
	    		imwrite("_C.png", anterior);
		}
     		imwrite("_N.png", nueva);
	}

	anterior=nueva.clone();	
	i++;


   }

    return 0;
} 

