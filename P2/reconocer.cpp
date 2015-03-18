/*
 * Autores: Aron Collados (626558)
 *          Cristian Roman (646564)
 */
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
    string image;
    if (argc == 1) {
        cout << "Introduza la ruta de la imagen;" << endl; //img/circulo2.pgm
        cin>> image;
    } else if (argc < 2) {
        cerr << argv[0] << " nomfich" << endl;
        return -1;
    } else {
        image = argv[1];
    }
    Mat bgrMap = imread(image, CV_LOAD_IMAGE_COLOR); //Carga la imagen recibida por parametro
    if (bgrMap.empty()) {
        cerr << "Could not open file " << image << endl;
        return -1;
    }
    vector<object>objs = getObjets();
    vector<vector<Point> > par = getContours(bgrMap);
    double alfa = 0;
    Mat m = identifyObject(bgrMap, par, objs, alfa);
    cout << "Pulsa 'escape' para salir" << endl;
    imshow("Objetos", m);
    char key = 0;
    while (key != 27) {
        imshow("Objetos", m);
        if (key == 32) { //space
            imwrite("detect.png", m);
        }
        if (key == 101) {//-
            alfa = alfa == 0.01 ? 0 : alfa - 0.01;
            m = identifyObject(bgrMap, par, objs, alfa);
            cout << alfa << endl;
        }
        if (key == 99) { //+
            alfa = alfa == -0.01 ? 0 : alfa + 0.01;
            m = identifyObject(bgrMap, par, objs, alfa);
            cout << alfa << endl;
        }
        key = waitKey(20);
    }
}
