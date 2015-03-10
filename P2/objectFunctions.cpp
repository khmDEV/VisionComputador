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
#include <math.h>
#include "fileSystem.h"
using namespace cv;
using namespace std;

Scalar color=Scalar( 0, 255, 0);
int font=FONT_HERSHEY_SIMPLEX;
float thicknessFont=1;

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
 * Filtro de ruido
 */
Mat removeNoise(Mat src) {
    Mat dst = src.clone();
    int v[3][3] = {{1, 2, 1},{2, 4, 2},{1, 2, 1} };
    //int v[5][5]={{1,4,6,4,1},{4,16,24,16,4},{6,24,36,24,6},{4,16,24,16,4},{1,4,6,4,1}};
    int m = 0;
    size_t r = sizeof (*v) / sizeof (*v[0]), c = r;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            m += v[i][j];
        }
    }

    int cr = r / 2, cc = c / 2;

    for (int i = cr; i < src.rows - cr; i++) {
        for (int j = cc; j < src.cols - cc; j++) {
            int tr = 0, tg = 0, tb = 0;
            for (int a = 0; a < r; a++) {
                for (int b = 0; b < c; b++) {
                    int coe = v[a][b];
                    tr += src.at<Vec3b>(i + a - cr, j + b - cc)[0] * coe;
                    tg += src.at<Vec3b>(i + a - cr, j + b - cc)[1] * coe;
                    tb += src.at<Vec3b>(i + a - cr, j + b - cc)[2] * coe;
                }
            }
            dst.at<Vec3b>(i, j)[0] = tr / m;
            dst.at<Vec3b>(i, j)[1] = tg / m;
            dst.at<Vec3b>(i, j)[2] = tb / m;
        }
    }
    return dst;
}
/*
 * Get contours
 */
pair<vector<vector<Point> >,vector<Vec4i> > getContours(Mat m){
   Mat binary=(Otsu(Grises(m)));
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;
   findContours( binary, contours, hierarchy,CV_RETR_LIST, CV_CHAIN_APPROX_TC89_L1, Point(0, 0) );
   pair<vector<vector<Point> >,vector<Vec4i> > p;
   p.first=contours;
   p.second=hierarchy;
   return p;
}
/*
 * Draw contours 
 */
Mat drawContors(Mat src,vector<vector<Point> > contours,vector<Vec4i> hierarchy) {
   	RNG rng(12345);//rand()*1000);//
  	Mat con = Mat::zeros( src.size(), CV_8UC3 );
   	for( int i = 0; i< contours.size(); i++ )
    	 {
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    		drawContours( con, contours, i, color, 2, 8, hierarchy, 0, Point() );
    	}
	return con;
}
/*
 * Calculate moments
 */
vector<Moments> calculateMoments(vector<vector<Point> > contours,int MINSIZE=0){
    vector<Moments> mu;

    for( int i = 0; i < contours.size(); i++ ){ 
	Moments m=moments( contours[i], false );
	if(MINSIZE<=m.m00){	
		mu.push_back(m); 
	}
    }
    return mu;
}

/*
 * Detect objects 
 */
Mat detectObject(Mat NuevaImagen,vector<vector<Point> > contours,int MINSIZE=1000){
	Mat out=NuevaImagen.clone();
	Size fontSize=getTextSize("1", font, 1, thicknessFont, 0);
        float scale=1.0/fontSize.width,MinSize=0.5;
	string str;
	char txt='0';
	vector<Moments> moms=calculateMoments(contours);
	for( int i = 0; i < contours.size(); i++ ){
		if(moms.at(i).m00>=MINSIZE){
			//Draw box
			RotatedRect rect = minAreaRect(contours.at(i));
			Point2f vertices[4];
			rect.points(vertices);
			for (int o = 0; o < 4; o++){
	   		 line(out, vertices[o], vertices[(o+1)%4], color);
			}
			//Draw text
			Point2f point=vertices[0];
			Mat txtMat=Mat::zeros( out.size(), CV_8UC3 );
			str=txt;//getType(mu[i]);
			float ss=scale*(rect.size.width/str.size());
			putText(txtMat, str, point, font, ss<MinSize?MinSize:ss,color, thicknessFont);
			float angle=abs((int)rect.angle)%180+(((int)rect.angle)-rect.angle);
			//Rotate text
	   		Mat r = getRotationMatrix2D(point, angle, 1.0);
	    		cv::warpAffine(txtMat, txtMat, r, txtMat.size());
			//NuevaImagen=NuevaImagen+txtMat;
			txtMat.copyTo(out, txtMat);
			txt++;
		}
	}
	return out;
}

/*
 * Calculate invariable moments
 */
vector<float> getMomentData(Moments m){
	vector<float> out;
	double arr[7];
        HuMoments(m, arr);
	for(int i=0;i<7;i++){
		out.push_back((float)arr[i]);
	}
	return out;
}

/*
 * Identify objects 
 */
float mahalanobis(object obt, vector<float> aln ){
    float aux = 0;
    for (int i=0;i<5;i++){
     aux= aux + pow((obt.mean.at(i)-aln.at(i)),2)/(obt.var.at(i));
    }
    return sqrt(aux);

}

string identifyObjectName(Moments m,vector<object> objs){
	vector<float> mi=getMomentData(m);
	string name="Unknow";
        float algo;
	for(int i=0;i<objs.size();i++){
		algo= mahalanobis(objs.at(i),mi);
                cout <<  algo << objs.at(i).name << endl;
	}
	return name;
}
Mat identifyObject(Mat NuevaImagen,vector<vector<Point> > contours,vector<object> objs,int MINSIZE=1000){
	Mat out=NuevaImagen.clone();
	Size fontSize=getTextSize("1", font, 1, thicknessFont, 0);
        float scale=1.0/fontSize.width,MinSize=0.5;
	string str;
	vector<Moments> moms=calculateMoments(contours);
	for( int i = 0; i < contours.size(); i++ ){
		if(moms.at(i).m00>=MINSIZE){
			//Draw box
			RotatedRect rect = minAreaRect(contours.at(i));		
			Point2f vertices[4];		
			rect.points(vertices);		
			for (int o = 0; o < 4; o++){
   				line(out, vertices[o], vertices[(o+1)%4], color);		
			}
			//Draw text
			Point2f point=vertices[0];		
			Mat txtMat=Mat::zeros( out.size(), CV_8UC3 );		
			str=identifyObjectName(moms.at(i),objs);//getType(mu[i]);
			float ss=scale*(rect.size.width/str.size());		
			putText(txtMat, str, point, font, ss<MinSize?MinSize:ss,color, thicknessFont);		
			float angle=abs((int)rect.angle)%180+(((int)rect.angle)-rect.angle);		
			//Rotate text
   			Mat r = getRotationMatrix2D(point, angle, 1.0);		
   	 		cv::warpAffine(txtMat, txtMat, r, txtMat.size());
			//NuevaImagen=NuevaImagen+txtMat;
			txtMat.copyTo(out, txtMat);
			cout<<"end"<<endl;
		}
	}
	return out;       
}



