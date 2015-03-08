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
#include "objectFunctions.h"

using namespace cv;
using namespace std;


/*
 * Main principal
 */
int main(int argc, char *argv[]) {
    if(argc<2){
    	cerr << argv[0] << " nomfich" << endl;
        return -1;
    }
    std::string image = argv[1];

    Mat bgrMap = imread(image, CV_LOAD_IMAGE_COLOR); //Carga la imagen recibida por parametro
    if (bgrMap.empty()) {
 	std::cerr << "Could not open file " << image << std::endl;
        return -1;
    }
   vector<object>objs= getObjets();
   pair<vector<vector<Point> >,vector<Vec4i> > par= getContours(bgrMap);
   Mat m=identifyObject(bgrMap,par.first,objs);
   cout << "Pulsa 'escape' para salir" << endl;
   namedWindow("Objetos", WINDOW_KEEPRATIO);
   imshow( "Objetos", m );
   char key='0';
   while (key != 27){ key = waitKey(20);}
}
