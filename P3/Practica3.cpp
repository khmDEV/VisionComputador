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

/*
 * Main principal
 */
int main(int argc, char *argv[]) {
    int MINSIZE;
    string obj,image;
    if (argc == 1) {
        cout << "Introduza la ruta de la imagen;" << endl; //img/circulo2.pgm
        cin>> image;
        cout << "Introduza el tipo de objeto" << endl;
        cin>> obj;
        MINSIZE = 1000;
    } else if (argc == 4) {
        MINSIZE = atoi(argv[4]);
    } else if (argc < 3) {
        cerr << " Error: nomfich nomobj [MINSIZE]" << endl;
        return -1;
    } else {
        MINSIZE = 1000;
    }

    Mat bgrMap = imread(image, CV_LOAD_IMAGE_COLOR); //Carga la imagen recibida por parametro
    if (bgrMap.empty()) {
        cerr << "Could not open file " << image << endl;
        return -1;
    }
    
    
    
       namedWindow( "Original image", CV_WINDOW_AUTOSIZE );
    imshow( "Original image", image );
 
    Mat grey;
    cvtColor(image, grey, CV_BGR2GRAY);
 
    Mat sobelx;
    Sobel(grey, sobelx, CV_32F, 1, 0);
 
    double minVal, maxVal;
    minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
    cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;
 
    Mat draw;
    sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
 
    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", draw);
 
    waitKey(0);                                        
    return 0;
} 
}