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
#include "fileSystem.h"
#include "objectFunctions.h"

using namespace cv;
using namespace std;

int scale = 1;
int delta = 0;
int ddepth = CV_32F;

/*
 * Main principal
 */
int main(int argc, char *argv[]) {
    string image;
    if (argc == 1) {
        cout << "Introduza la ruta de la imagen;" << endl; //img/circulo2.pgm
        cin>> image;
    } else {
        image = argv[1];
    }

    Mat bgrMap = imread(image, CV_LOAD_IMAGE_COLOR); //Carga la imagen recibida por parametro
    if (bgrMap.empty()) {
        cerr << "Could not open file " << image << endl;
        return -1;
    }



    namedWindow("Original image", CV_WINDOW_AUTOSIZE);
    namedWindow("Gx", CV_WINDOW_AUTOSIZE);
    namedWindow("Magnitud del Gradiente", CV_WINDOW_AUTOSIZE);
    imshow("Original image", bgrMap);


    GaussianBlur(bgrMap, bgrMap, Size(3, 3), 0, 0, BORDER_DEFAULT); //Filtro Gausiano

    Mat grey;
    cvtColor(bgrMap, grey, CV_BGR2GRAY);


    //Gradiente X////////////////////////////////////
    Mat sobelx;
    Sobel(grey, sobelx, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);

    double minValX, maxValX;
    minMaxLoc(sobelx, &minValX, &maxValX); //find minimum and maximum intensities
    cout << "Gx" << "minVal : " << minValX << endl << "maxVal : " << maxValX << endl;

    Mat drawX;
    sobelx.convertTo(drawX, CV_8U, 255.0 / (maxValX - minValX), -minValX * 255.0 / (maxValX - minValX));

    imshow("Gx", drawX);

    //////////////////////////////////////////////////////////


    //Gradiente Y///////////////////////////////////////
    Mat sobely;
    Sobel(grey, sobely, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    double minValY, maxValY;
    minMaxLoc(sobely, &maxValY, &minValY); //find minimum and maximum intensities
    cout << "Gy" << "minVal : " << minValY << endl << "maxVal : " << maxValY << endl;

    Mat drawY;
    sobely.convertTo(drawY, CV_8U, 255.0 / (maxValY - minValY), -minValY * 255.0 / (maxValY - minValY));

    imshow("Gy", drawY);
    //////////////////////////////////////////////////


    //Magnitud del Gradiente////////////////
    Mat grad;
    Mat abs_grad_x, abs_grad_y;
    convertScaleAbs(sobelx, abs_grad_x);
    convertScaleAbs(sobely, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    imshow("Magnitud del Gradiente", grad);

    /////////////////////////////////////////////////


    waitKey(0);
    return 0;
} 
