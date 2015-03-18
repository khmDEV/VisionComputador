/*
 * Autores: Aron Collados (626558)
 *          Cristian Roman (646564)
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <math.h>
#include "fileSystem.h"
using namespace cv;
using namespace std;

const Scalar color = Scalar(0, 255, 0);
const int font = FONT_HERSHEY_SIMPLEX;
const float thicknessFont = 1, MARGEN = 11.07;

/*
 * Grises
 */
Mat Grises(Mat bgrMap) {
    Mat dst;
    cvtColor(bgrMap, dst, CV_RGB2GRAY);
    return dst;
}

/*
 * Otsu effect
 */
Mat Otsu(Mat src) {
    Mat dst;
    threshold(src, dst, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
    return dst;
}

/*
 * Adatative effect
 */
Mat adaptative(Mat src) {
    Mat dst;
    adaptiveThreshold(src, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 75, 10);
    return dst;
}

/*
 * Get contours
 */
vector<vector<Point> > getContours(Mat m,int MINSIZE=1000) {
    Mat binary = (Otsu(Grises(m)));
    vector<vector<Point> > contours,out;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));
    for(int i=0;i<contours.size();i++){
        Moments m = moments(contours.at(i), false);
        if (MINSIZE <= m.m00) {
            out.push_back(contours.at(i));
        }
    }
    return out;
}

/*
 * Draw contours 
 */
Mat drawContors(Mat src, vector<vector<Point> > contours) {
    RNG rng(12345); //rand()*1000);//
    Mat con = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(con, contours, i, color, 2, 8);
    }
    return con;
}

/*
 * Calculate moments
 */
vector<Moments> calculateMoments(vector<vector<Point> > contours, int MINSIZE = 0) {
    vector<Moments> mu;

    for (int i = 0; i < contours.size(); i++) {
        Moments m = moments(contours[i], false);
        if (MINSIZE <= m.m00) {
            mu.push_back(m);
        }
    }
    return mu;
}

/*
 * Detect objects 
 */
Mat detectObject(Mat NuevaImagen, vector<vector<Point> > contours, int MINSIZE = 1000) {
    Mat out = NuevaImagen.clone();
    Size fontSize = getTextSize("1", font, 1, thicknessFont, 0);
    float scale = 1.0 / fontSize.width, MinSize = 0.5;
    string str;
    char txt = '0';
    vector<Moments> moms = calculateMoments(contours);
  
    for (int i = 0; i < contours.size(); i++) {
        if (moms.at(i).m00 >= MINSIZE) {
            //Draw box
            RotatedRect rect = minAreaRect(contours.at(i));
            Point2f vertices[4];
            rect.points(vertices);
            for (int o = 0; o < 4; o++) {
                line(out, vertices[o], vertices[(o + 1) % 4], color);
            }
            //Draw text
            Point2f point = vertices[0];
            Mat txtMat = Mat::zeros(out.size(), CV_8UC3);
            str = txt; //getType(mu[i]);
            float ss = scale * (rect.size.width / str.size());
            putText(txtMat, str, point, font, ss < MinSize ? MinSize : ss, color, thicknessFont);
            float angle = abs((int) rect.angle) % 180 + (((int) rect.angle) - rect.angle);
            //Rotate text
            Mat r = getRotationMatrix2D(point, angle, 1.0);
            warpAffine(txtMat, txtMat, r, txtMat.size());
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
vector<float> getMomentData(Moments m) {
    vector<float> out;
    double arr[7];
    HuMoments(m, arr);
    for (int i = 0; i < 7; i++) {
        out.push_back((float) arr[i]);
    }
    return out;
}

/*
 * Identify objects 
 */
float mahalanobis(object obt, vector<float> aln,double alfa) {
    float aux = 0;
    int n= 5;
      float devreg ;
    for (int i = 0; i < 5; i++) {
        if(alfa!=0){
            devreg   = (pow(obt.mean.at(i)*alfa, 2.0) / n)+ ((n - 1 / n) * obt.var.at(i));
            aux = aux + pow((obt.mean.at(i) - aln.at(i)), 2.0) / (devreg);
        }else{
            aux = aux + pow((obt.mean.at(i) - aln.at(i)), 2.0) / (obt.var.at(i));
        }
    }
    return sqrt(aux);

}

string identifyObjectName(Moments m,int perim, vector<object> objs, double alfa) {
    vector<float> mi = getMomentData(m);
    
    vector<float> cmp;
    cmp.push_back(m.m00);
    cmp.push_back(perim);
    for (int i = 0; i < mi.size(); i++) {
        cmp.push_back(mi.at(i));
    }
    for (int i = 0; i < cmp.size(); i++) {
        cout << "<<<<< "<< cmp.at(i)<<endl;
    }
    string name = "Unknow";
    float algo, n = 0; //,f;
    for (int i = 0; i < objs.size(); i++) {
        algo = mahalanobis(objs.at(i), cmp,alfa);
        cout << algo << objs.at(i).name << endl;
        if (MARGEN > algo) {
            n++;
            if (n < 2) {
                name = objs.at(i).name;
            } else if (n > 1) {
                return "Doubfull";
            }
        }
    }
    return name;
}

Mat identifyObject(Mat NuevaImagen, vector<vector<Point> > contours, vector<object> objs,double alfa, int MINSIZE = 1000) {
    Mat out = NuevaImagen.clone();
    Size fontSize = getTextSize("1", font, 1, thicknessFont, 0);
    float scale = 1.0 / fontSize.width, MinSize = 0.5;
    string str;
    vector<Moments> moms = calculateMoments(contours);
    for (int i = 0; i < contours.size(); i++) {
        if (moms.at(i).m00 >= MINSIZE) {
            //Draw box
            RotatedRect rect = minAreaRect(contours.at(i));
            Point2f vertices[4];
            rect.points(vertices);
            for (int o = 0; o < 4; o++) {
                line(out, vertices[o], vertices[(o + 1) % 4], color);
            }
            //Draw text
            Point2f point = vertices[0];
            Mat txtMat = Mat::zeros(out.size(), CV_8UC3);
            double perim=arcLength(contours.at(i),true);
            str = identifyObjectName(moms.at(i),perim, objs, alfa); //getType(mu[i]);
           // if(str=="Doubfull"&&alfa!=0){
              //  str = identifyObjectName(moms.at(i),perim, objs, alfa); //getType(mu[i]);
            //}
            float ss = scale * (rect.size.width / str.size());
            putText(txtMat, str, point, font, ss < MinSize ? MinSize : ss, color, thicknessFont);
            float angle = abs((int) rect.angle) % 180 + (((int) rect.angle) - rect.angle);
            //Rotate text
            Mat r = getRotationMatrix2D(point, angle, 1.0);
            warpAffine(txtMat, txtMat, r, txtMat.size());
            //NuevaImagen=NuevaImagen+txtMat;
            txtMat.copyTo(out, txtMat);
            cout << "end" << endl;
        }
    }
    return out;
}



