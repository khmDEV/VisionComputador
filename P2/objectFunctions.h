/*
 * Autores: Aron Collados (626558)
 *          Cristian Roman (646564)
 */
#ifndef OBJECTFUNTIONS_H
#define OBJECTFUNTIONS_H

#include <vector>
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

Mat Grises(Mat bgrMap);
Mat Otsu(Mat src);
Mat adaptative(Mat src);
Mat removeNoise(Mat src);
vector<vector<Point> > getContours(Mat m);
Mat drawContors(Mat src,vector<vector<Point> > contours);
vector<Moments> calculateMoments(vector<vector<Point> >,int MINSIZE=0);
Mat detectObject(Mat NuevaImagen,vector<vector<Point> >,int MINSIZE=1000);
vector<float> getMomentData(Moments);
Mat identifyObject(Mat NuevaImagen,vector<vector<Point> >,vector<object>,double alfa, int MINSIZE=1000);

#endif
