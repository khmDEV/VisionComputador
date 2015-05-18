/*
* Autores: Aron Collados (626558)
*          Cristian Roman (646564)
*/
#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace cv;
using namespace std;

int MIN_MATCHES = 10;
int WINDOWS_TYPE = CV_WINDOW_NORMAL;
double GOOD_DISTANCE = 0.02;
int MIN_KEYPOINTS = 8;
string CALIBRATION_FILE = "calibration.yml";
Mat img_matches;
int metodoP = 0;
int MAX_SIZE=1000;
int DIV=5;
/*
* Calcula las coordenadas resultantes de un punto
* al aplicarle una Homografia
*/
pair<double, double> calculateCoor(Mat H, int x, int y) {
  pair<double, double> p;
  double xC = H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2);
  double yC = H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2);
  double zC = H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2);
  p.first = round(xC / zC);
  p.second = round(yC / zC);
  return p;
}

/*
* Calcula el offset necesario para que la imagen no tenga pixeles
* en coordenadas negativas y si alguno de los valores sea atipico
* devuelve este
*/
pair<int, int> calculateOffset(Mat H, Mat img) {

  pair<double, double> A = calculateCoor(H, 0, 0);
  pair<double, double> B = calculateCoor(H, 0, img.rows);
  pair<double, double> C = calculateCoor(H, img.cols, 0);
  pair<double, double> D = calculateCoor(H, img.cols, img.rows);

  double minX = min(min(A.first, B.first), min(C.first, D.first));
  double minY = min(min(A.second, B.second), min(C.second, D.second));

  pair<double, double> p;
  p.first = minX < 0 ? abs(minX) : 0;
  p.second = minY < 0 ? abs(minY) : 0;

  return p;
}

/*
* Aplica un offset a una homografia
*/
void applyOffset(Mat m, double x, double y) {
  Mat f = Mat::eye(3, 3, m.type());
  f.at<double>(0, 2) = x;
  f.at<double>(1, 2) = y;
  Mat R = f*m;
  R.copyTo(m);
  R.release();
  f.release();
}

/*
* Recorta el fondo negro de una imagen
*/
void crop(Mat &in) {
  vector<cv::Point> nonBlackList;
  nonBlackList.reserve(in.rows * in.cols);

  for (int j = 0; j < in.rows; ++j) {
    for (int i = 0; i < in.cols; ++i) {
      if (in.at<cv::Vec3b>(j, i) != Vec3b(0, 0, 0)) {
        nonBlackList.push_back(cv::Point(i, j));
      }
    }
  }
  Rect bb = cv::boundingRect(nonBlackList);
  Mat crop = in(bb);
  crop.copyTo(in);
  crop.release();
}



void comparar(Mat original, Mat next){
  Mat originalG, nextG, img_matchesc;

  /*
  * Tranformamos las imagenes a grises
  */
  cvtColor(original, originalG, CV_BGR2GRAY);
  cvtColor(next, nextG, CV_BGR2GRAY);

  /*
  * Detecion de keypoints
  */

  vector<KeyPoint> keypoints1, keypoints2;
  if (metodoP == 0) {
    SiftFeatureDetector detector;
    detector.detect(originalG, keypoints1);
    detector.detect(nextG, keypoints2);
  } else if (metodoP == 1) {
    SurfFeatureDetector detector(400); //400 = minHessian
    detector.detect(originalG, keypoints1);
    detector.detect(nextG, keypoints2);
  } else if (metodoP == 2) {
    OrbFeatureDetector detector;
    detector.detect(originalG, keypoints1);
    detector.detect(nextG, keypoints2);
  } else {
    FastFeatureDetector detector;
    detector.detect(originalG, keypoints1);
    detector.detect(nextG, keypoints2);
  }

  if (keypoints2.size() < MIN_KEYPOINTS) {
    cout << "La imagen original no tiene suficientes keypoints" << endl;
    originalG.release();
    nextG.release(); //liberamos los mat
  } else if (keypoints1.size() < MIN_KEYPOINTS) {
    cout << "La imagen comparada no tiene suficientes keypoints" << endl;
    originalG.release();
    nextG.release(); //liberamos los mat
  }
  /*
  * Extractor de descriptores
  */

  Mat descriptors1, descriptors2;
  if (metodoP == 0) {
    SiftDescriptorExtractor extractor;
    extractor.compute(originalG, keypoints1, descriptors1);
    extractor.compute(nextG, keypoints2, descriptors2);
  } else if (metodoP == 1) {
    SurfDescriptorExtractor extractor;
    extractor.compute(originalG, keypoints1, descriptors1);
    extractor.compute(nextG, keypoints2, descriptors2);
  } else if (metodoP == 2) {
    OrbDescriptorExtractor extractor;
    extractor.compute(originalG, keypoints1, descriptors1);
    extractor.compute(nextG, keypoints2, descriptors2);
  } else {
    SurfDescriptorExtractor extractor;
    extractor.compute(originalG, keypoints1, descriptors1);
    extractor.compute(nextG, keypoints2, descriptors2);
  }

  originalG.release();
  nextG.release(); //liberamos los mat
  /*
  * Matching de keypoints
  */
  vector<DMatch> matches;
  if (metodoP == 0) {
    FlannBasedMatcher matcher;
    matcher.match(descriptors1, descriptors2, matches);
  } else {
    BFMatcher matcher(NORM_L2);
    matcher.match(descriptors1, descriptors2, matches);
  }



  /*
  * Calculo de distancia optima
  */
  double max_dist = 0;
  double min_dist = 100;

  for (int i = 0; i < descriptors1.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }


  /*
  * Calculo de matches buenos
  */
  vector< DMatch > good_matches;
  for (int i = 0; i < descriptors1.rows; i++) {
    if (matches[i].distance <= max(3 * min_dist, GOOD_DISTANCE)) {
      good_matches.push_back(matches[i]);
    }
  }


  descriptors1.release();
  descriptors2.release(); //liberamos los mat

  if (good_matches.size() < MIN_KEYPOINTS) {
    cout << "No se han encontrado suficiente matches buenos" << endl;
    drawMatches(original, keypoints1, next, keypoints2, matches, img_matchesc);
    if (original.rows == next.rows && original.cols == next.cols) {
      cout << "Se seguira utilizando la imagen comparada" << endl;
    }
    cout << "Se seguira utilizando la imagen original" << endl;
  }

  vector< Point2f > obj;
  vector< Point2f > scene;
  for (int i = 0; i < good_matches.size(); i++) {
    obj.push_back(keypoints1[ good_matches[i].queryIdx ].pt);
    scene.push_back(keypoints2[ good_matches[i].trainIdx ].pt);
  }

  drawMatches(original, keypoints1, next, keypoints2, good_matches, img_matchesc, Scalar::all(-1), Scalar::all(-1),
  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


  /*
  * Calcula la homografia
  */
  //    findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
  Mat H = findHomography(obj, scene, CV_RANSAC);
  if (H.at<double>(1, 0) > 0.25 || H.at<double>(1, 0)<-0.25 || H.at<double>(0, 0) < 0.75 || H.at<double>(0, 0) > 1.25) {
    cout << "Imagen descartada: Homografia atipica" << endl;
  }

  /*
  * Representacion emparejamiento de puntos
  */
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0, 0);
  obj_corners[1] = cvPoint(original.cols, 0);
  obj_corners[2] = cvPoint(original.cols, original.rows);
  obj_corners[3] = cvPoint(0, original.rows);
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform(obj_corners, scene_corners, H);

  line(img_matchesc, scene_corners[0] + Point2f(original.cols, 0), scene_corners[1] + Point2f(original.cols, 0), Scalar(0, 255, 0), 4);
  line(img_matchesc, scene_corners[1] + Point2f(original.cols, 0), scene_corners[2] + Point2f(original.cols, 0), Scalar(0, 255, 0), 4);
  line(img_matchesc,scene_corners[2] + Point2f(original.cols, 0), scene_corners[3] + Point2f(original.cols, 0), Scalar(0, 255, 0), 4);
  line(img_matchesc, scene_corners[3] + Point2f(original.cols, 0), scene_corners[0] + Point2f(original.cols, 0), Scalar(0, 255, 0), 4);


   imshow("Coincidencias", img_matchesc);



}





Mat mount(Mat original, Mat next) {
  Mat originalG, nextG;

  /*
  * Tranformamos las imagenes a grises
  */
  cvtColor(original, originalG, CV_BGR2GRAY);
  cvtColor(next, nextG, CV_BGR2GRAY);

  /*
  * Detecion de keypoints
  */

  vector<KeyPoint> keypoints1, keypoints2;
  if (metodoP == 0) {
    SiftFeatureDetector detector;
    detector.detect(originalG, keypoints1);
    detector.detect(nextG, keypoints2);
  } else if (metodoP == 1) {
    SurfFeatureDetector detector(400); //400 = minHessian
    detector.detect(originalG, keypoints1);
    detector.detect(nextG, keypoints2);
  } else if (metodoP == 2) {
    OrbFeatureDetector detector;
    detector.detect(originalG, keypoints1);
    detector.detect(nextG, keypoints2);
  } else {
    FastFeatureDetector detector;
    detector.detect(originalG, keypoints1);
    detector.detect(nextG, keypoints2);
  }

  if (keypoints2.size() < MIN_KEYPOINTS) {
    cout << "La imagen original no tiene suficientes keypoints" << endl;
    originalG.release();
    nextG.release(); //liberamos los mat
    return original;
  } else if (keypoints1.size() < MIN_KEYPOINTS) {
    cout << "La imagen comparada no tiene suficientes keypoints" << endl;
    originalG.release();
    nextG.release(); //liberamos los mat
    return next;
  }
  /*
  * Extractor de descriptores
  */

  Mat descriptors1, descriptors2;
  if (metodoP == 0) {
    SiftDescriptorExtractor extractor;
    extractor.compute(originalG, keypoints1, descriptors1);
    extractor.compute(nextG, keypoints2, descriptors2);
  } else if (metodoP == 1) {
    SurfDescriptorExtractor extractor;
    extractor.compute(originalG, keypoints1, descriptors1);
    extractor.compute(nextG, keypoints2, descriptors2);
  } else if (metodoP == 2) {
    OrbDescriptorExtractor extractor;
    extractor.compute(originalG, keypoints1, descriptors1);
    extractor.compute(nextG, keypoints2, descriptors2);
  } else {
    SurfDescriptorExtractor extractor;
    extractor.compute(originalG, keypoints1, descriptors1);
    extractor.compute(nextG, keypoints2, descriptors2);
  }

  originalG.release();
  nextG.release(); //liberamos los mat
  /*
  * Matching de keypoints
  */
  vector<DMatch> matches;
  if (metodoP == 0 && metodoP == 3) {
    FlannBasedMatcher matcher;
    matcher.match(descriptors1, descriptors2, matches);
  } else {
    BFMatcher matcher(NORM_L2);
    matcher.match(descriptors1, descriptors2, matches);
  }



  /*
  * Calculo de distancia optima
  */
  double max_dist = 0;
  double min_dist = 100;

  for (int i = 0; i < descriptors1.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }


  /*
  * Calculo de matches buenos
  */
  vector< DMatch > good_matches;
  for (int i = 0; i < descriptors1.rows; i++) {
    if (matches[i].distance <= max(3 * min_dist, GOOD_DISTANCE)) {
      good_matches.push_back(matches[i]);
    }
  }


  descriptors1.release();
  descriptors2.release(); //liberamos los mat

  if (good_matches.size() < MIN_KEYPOINTS) {
    cout << "No se han encontrado suficiente matches buenos" << endl;
    drawMatches(original, keypoints1, next, keypoints2, matches, img_matches);
    if (original.rows == next.rows && original.cols == next.cols) {
      cout << "Se seguira utilizando la imagen comparada" << endl;
      return next;
    }
    cout << "Se seguira utilizando la imagen original" << endl;
    return original;
  }

  vector< Point2f > obj;
  vector< Point2f > scene;
  for (int i = 0; i < good_matches.size(); i++) {
    obj.push_back(keypoints1[ good_matches[i].queryIdx ].pt);
    scene.push_back(keypoints2[ good_matches[i].trainIdx ].pt);
  }

  drawMatches(original, keypoints1, next, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  /*
  * Calcula la homografia
  */
  //    findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
  Mat H = findHomography(obj, scene, CV_RANSAC);
  if (H.at<double>(1, 0) > 0.25 || H.at<double>(1, 0)<-0.25 || H.at<double>(0, 0) < 0.75 || H.at<double>(0, 0) > 1.25) {
    cout << "Imagen descartada: Homografia atipica" << endl;
    return original;
  }

  /*
  * Representacion emparejamiento de puntos
  */
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0, 0);
  obj_corners[1] = cvPoint(original.cols, 0);
  obj_corners[2] = cvPoint(original.cols, original.rows);
  obj_corners[3] = cvPoint(0, original.rows);
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform(obj_corners, scene_corners, H);

  line(img_matches, scene_corners[0] + Point2f(original.cols, 0), scene_corners[1] + Point2f(original.cols, 0), Scalar(0, 255, 0), 4);
  line(img_matches, scene_corners[1] + Point2f(original.cols, 0), scene_corners[2] + Point2f(original.cols, 0), Scalar(0, 255, 0), 4);
  line(img_matches, scene_corners[2] + Point2f(original.cols, 0), scene_corners[3] + Point2f(original.cols, 0), Scalar(0, 255, 0), 4);
  line(img_matches, scene_corners[3] + Point2f(original.cols, 0), scene_corners[0] + Point2f(original.cols, 0), Scalar(0, 255, 0), 4);




  /*
  * Fix offset
  */
  pair<double, double> off = calculateOffset(H, original);
  applyOffset(H, off.first, off.second);

  /*
  * Obtiene imagen final
  */
  Mat nueva(original.rows + next.rows + off.first, original.cols + next.cols + off.second, original.type(), Scalar(0, 0, 0));
  //warpPerspective(original, nueva, H, nueva.size(), INTER_CUBIC);
  warpPerspective(original, nueva, H, nueva.size());
  Mat half(nueva, cv::Rect(off.first, off.second
    , next.cols, next.rows));
    next.copyTo(half);

    crop(nueva); //Recorta imagen
    half.release();
    H.release();
    return nueva;
  }

  /*
  * Main principal
  */
  int main(int argc, char *argv[]) {
    char key = 0;
    string image;
    string num;
    int i = 1;
    int cap=0;
    int init,end,mean;
    clock_t prevTimestamp = 0;
    bool notStop = false;
    VideoCapture TheVideoCapturer;
    Mat captura, nueva, anterior;
    namedWindow("Original", WINDOWS_TYPE);
    namedWindow("Comparacion", WINDOWS_TYPE);
    namedWindow("matches", WINDOWS_TYPE);
    namedWindow("Nueva", WINDOWS_TYPE);

    /*
    * Carga datos de calibracion
    */
    bool calibrate = false;
    Mat map1, map2;


    fstream file(CALIBRATION_FILE.c_str());


    if (file.good()) {
      cout << "La camara esta calibrada" << endl;

      Mat cameraMatrix, distCoeffs;
      Size imageSize;

      calibrate = true;
      FileStorage fs(CALIBRATION_FILE.c_str(), FileStorage::READ);
      fs["Camera_Matrix"] >> cameraMatrix;
      fs["Distortion_Coefficients"] >> distCoeffs;
      initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
      getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
      imageSize, CV_16SC2, map1, map2);

      fs.release();
      cameraMatrix.release();
      distCoeffs.release();
    }

    file.close();
    cout << "Metodo SIFT"<<endl;
    cout << "////////////////////////////////////////" << endl;
    cout << "Pulsa m para cambiar el modo" << endl;
    cout << "Pulsa otra tecla cualquiera para contcinuar" << endl;
    cout << "////////////////////////////////////////" << endl;
    key = waitKey(0);
    while (key == 109) {
        metodoP = metodoP == 3 ? 0 : metodoP + 1;
        
        cout << "Metodo ";
	if(metodoP==0){
		cout << "SIFT";
	}else if(metodoP==1){
		cout << "SURF";
	}else if(metodoP==2){
		cout << "ORB";
	}else if(metodoP==3){
		cout << "FAST";
	}
	cout << endl;
        key = waitKey(0);
    }
    do {
      if (!TheVideoCapturer.isOpened()) {
        if (argc <= i) {
          cout << "////////////////////////////////////////" << endl;
          cout << "Pulsa esc para salir" << endl;
          cout << "Pulsa otra tecla cualquiera para contcinuar" << endl;
          cout << "////////////////////////////////////////" << endl;
          key = waitKey(0);
          if (key == 27) {
            return 0;
          }
          cout << "Introduza la ruta de la imagen;" << endl;
          cin>> image;
        } else {
          image = argv[i];
        }
      }
      captura.release();
      captura = imread(image, CV_LOAD_IMAGE_COLOR); //Carga la imagen recibida por parametro
	
      if (captura.empty()&&!TheVideoCapturer.isOpened()) {
        TheVideoCapturer.open(atoi(image.c_str())); //Abre la videocamara
      }
      if (!TheVideoCapturer.isOpened() && captura.empty()) {
        std::cerr << "Could not open file " << image << std::endl;
        return -1;
      }
      if (captura.empty()) {
        captura.release();
        TheVideoCapturer >> captura;
      }

      if(captura.rows>MAX_SIZE||captura.cols>MAX_SIZE){
      	Size siz=Size(captura.cols/DIV,captura.rows/DIV);
	while(siz.height>MAX_SIZE||siz.width>MAX_SIZE){
		siz=Size(siz.height/DIV,siz.width/DIV);
	}
      	Mat aux=captura;
	resize(aux,captura,siz);
	aux.release();
      }

      if (!anterior.empty()) {

        nueva.release();
	init=clock();
        nueva = mount(anterior, captura);
	end=clock();
	int time=end-init;
	cout<<"Tiempo "	<<time<< " ms"<<endl;
        imshow("Original", anterior);
        imshow("Nueva", nueva);
        if (!img_matches.empty()) {
          imshow("matches", img_matches);
        }
        imshow("Comparacion", captura);
      } else {
        nueva = captura;
        imshow("Original", captura);
      }

      if (!notStop) {
        cout << "////////////////////////////////////////" << endl;
        cout << "Pulsa esc para salir" << endl;
        cout << "Pulsa espacio para guardar las capturas" << endl;
        cout << "Pulsa enter para obtener imagenes automaticamente" << endl;
        cout << "Pulsa m para cambiar el modo" << endl;
        cout << "Pulsa otra tecla cualquiera para contcinuar" << endl;
        cout << "////////////////////////////////////////" << endl;
        key = waitKey(0);
        if (key == 99) {
          notStop = true;
        }
      } else {
        key = waitKey(1);
      }
      if (key == 109) {
        metodoP = metodoP == 3 ? 0 : metodoP + 1;
        cout << "Metodo" << metodoP << endl;
      }
      if (key == 32) {
      	stringstream ss;
      	ss<<cap;
	ss>>num;
        imwrite(num+"_F.png", captura);
        imwrite(num+"_C.png", anterior);
        imwrite(num+"_N.png", nueva);
        imwrite(num+"_M.png", img_matches);
	cap++;
      }

      if(key == 104){ //h
        Mat imagenA, imagenB;
        cout << "Introduza la ruta de la imagen1;" << endl;
        cin>> image;
        imagenA = imread(image, CV_LOAD_IMAGE_COLOR);
        cout << "Introduza la ruta de la imagen2;" << endl;
        cin>> image;
        imagenB = imread(image, CV_LOAD_IMAGE_COLOR);
        if (!imagenA.empty() && !imagenB.empty()){
          comparar(imagenA,imagenB);
        }
      }
      anterior.release();
      anterior = nueva;
      i++;
    } while (key != 27);
    return 0;
  }
  
