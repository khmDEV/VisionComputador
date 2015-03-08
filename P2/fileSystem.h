/*
 * Autores: Aron Collados (626558)
 *          Cristian Roman (646564)
 */
#ifndef FILESYSTEM_H
#define FILESYSTEM_H
#include <vector>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

struct object;

bool addMoment(const char*,Moments);
vector<object> getObjets();

#endif
