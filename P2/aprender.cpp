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
    if(argc<3){
    	cerr << argv[0] << " nomfich nomobj [MINSIZE]" << endl;
        return -1;
    }
    std::string image = argv[1],obj = argv[2],size=argc>=4?argv[3]:"1000";
    int MINSIZE=atoi(size.c_str());

    Mat bgrMap = imread(image, CV_LOAD_IMAGE_COLOR); //Carga la imagen recibida por parametro
    if (bgrMap.empty()) {
 	std::cerr << "Could not open file " << image << std::endl;
        return -1;
    }
    /*
     * Get Contors
     */
   pair<vector<vector<Point> >,vector<Vec4i> > par= getContours(bgrMap);
   
    /*
     * Calculate Moments
     */
    vector<Moments> mu=calculateMoments(par.first,MINSIZE);
    Moments mS;
    if(mu.size()>1){
	Mat mat=detectObject(bgrMap,par.first,MINSIZE);
    //namedWindow("Objetos",  WINDOW_KEEPRATIO);
	imshow("Objetos", mat);
	waitKey(20);//Soluciona problema por el cual no se mostraba la imagen
        int i;
	do{
		cout << "Select the object:"<<endl;
		cin >> i;
	}while(i<0||i>=mu.size());
        mS=mu.at(i);
    }else if(mu.size()<1){
    	cerr << "Error: Object not found with minimum size: " << MINSIZE << endl;
	return -1;
    }else{
    	mS=mu.at(0);
    }
    /*
     * Learn
     */
    addMoment(obj.c_str(),mS);
    cout << "Aprendido!"<<endl;
}
