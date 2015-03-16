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
    string obj;
    char * image;
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
    /*
     * Get Contors
     */
    vector<vector<Point> > par = getContours(bgrMap);

    /*
     * Calculate Moments
     */
    vector<Moments> mu = calculateMoments(par, MINSIZE);
    Moments mS;
    if (mu.size() > 1) {
        Mat mat = detectObject(bgrMap, par, MINSIZE);
        imshow("Objetos", mat);
        waitKey(20); //Soluciona problema por el cual no se mostraba la imagen
        int i;
        do {
            cout << "Select the object:" << endl;
            cin >> i;
        } while (i < 0 || i >= mu.size());
        mS = mu.at(i);
    } else if (mu.size() < 1) {
        cerr << "Error: Object not found with minimum size: " << MINSIZE << endl;
        return -1;
    } else {
        mS = mu.at(0);
    }
    /*
     * Learn
     */
    addMoment(obj.c_str(), mS);
    cout << "Aprendido!" << endl;
}
