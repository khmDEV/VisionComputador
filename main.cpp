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

bool test = true,noise = false;
double alpha, beta; 
float cof = 1;
float correctorX = 1.33, correctorY = 1.33;
int filtro, // 0-Nornal, 1- Filtro de grises, 2-Escala de colores,3-Filtro alienacion, 4- Filtro Negativo
	alienMode=0;
const Vec3b black(0, 0, 0),white(255, 255, 255),green = (0, 0, 255);

Mat contrasteRGB(Mat image) {
    Mat nuevaImagen = Mat::zeros(image.size(), image.type());

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int c = 0; c < 3; c++) { //RGB
                nuevaImagen.at<Vec3b>(y, x)[c] =
                        saturate_cast<uchar>(alpha * (image.at<Vec3b>(y, x)[c]) + beta); //x= alpha *x + beta
            }
        }
    }
    return nuevaImagen;
}

Mat contrasteHSI(Mat image) {
    Mat newimage;
    cvtColor(image, newimage, CV_BGR2HSV);

    Mat nuevaImagen = Mat::zeros(image.size(), newimage.type());

    for (int y = 0; y < newimage.rows; y++) {
        for (int x = 0; x < newimage.cols; x++) {
            nuevaImagen.at<Vec3b>(y, x)[2] =
                    saturate_cast<uchar>(alpha * (newimage.at<Vec3b>(y, x)[2])); //x= alpha *x + beta  
            nuevaImagen.at<Vec3b>(y, x)[0] = newimage.at<Vec3b>(y, x)[0];
            nuevaImagen.at<Vec3b>(y, x)[1] = newimage.at<Vec3b>(y, x)[1];
        }
    }
    cvtColor(nuevaImagen, nuevaImagen, CV_HSV2BGR);
    return nuevaImagen;
}

Mat barrel(Mat imagen, double Cx, double Cy, double kx, double ky) {
    Mat dst = Mat::zeros(imagen.size(), imagen.type());
    Mat mapx = Mat::zeros(imagen.size(), CV_32FC1);
    Mat mapy = Mat::zeros(imagen.size(), CV_32FC1);
    int h = imagen.rows;
    int w = imagen.cols;
    double rTot = sqrt(Cx * Cx + Cy * Cy);
    for (int y = 0; y < h; y++) {
        int ty = y - Cy;
        for (int x = 0; x < w; x++) {
            int tx = x - Cx;
            float rt = sqrt(tx * tx + ty * ty) / rTot;
            mapx.at<float>(y, x) = (float) (tx * (1 + kx * rt * rt) * correctorX + Cx);
            mapy.at<float>(y, x) = (float) (ty * (1 + ky * rt * rt) * correctorY + Cy);
        }
    }
    remap(imagen, dst, mapx, mapy, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    return dst;
}

bool R1(int R, int G, int B) { //Mismos cooeficientes RGB
    bool e1 = (R > 95) && (G > 40) && (B > 20) ;//&& ((max(R, max(G, B)) - min(R, min(G, B))) > 15) && (abs(R - G) > 15) && (R > G) && (R > B);
    bool e2 = (R > 220) && (G > 210) && (B > 170) ;//&& (abs(R - G) <= 15) && (R > B) && (G > B);
    return (e1 || e2);
}

bool R2(float Y, float Cr, float Cb) { //Coenficientes YCrCb
    bool e3 = Cr <= 1.5862 * Cb + 20;
    bool e4 = Cr >= 0.3448 * Cb + 76.2069;
    bool e5 = Cr >= -4.5652 * Cb + 234.5652;
    bool e6 = Cr <= -1.15 * Cb + 301.75;
    bool e7 = Cr <= -2.2857 * Cb + 432.85;
    return e3 && e4 && e5 && e6 && e7;
}

bool R2A(float Y, float Cr, float Cb) { //Coenficientes YCrCb
    return ((Y > 80) && ((Cb > 85) || (Cr < 135)) && ((Cr > 135) || (Cr < 180)));
}

bool R3(float H, float S, float V) { //Coeficientes HSV
    //return (H < 25) || (H > 230);
    return (((H < 25) || (H > 230)) && ((S > 10) || (S < 150)) && (V > 60));
}

Mat alien(Mat image) { //Detectar piel escala RGB
    Mat nuevaImagen = image.clone();
    int R, G, B;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            R = image.at<Vec3b>(y, x)[0];
            G = image.at<Vec3b>(y, x)[1];
            B = image.at<Vec3b>(y, x)[2];

            if (R1(R, G, B)) {
                nuevaImagen.ptr<Vec3b>(y)[x] = green;
            }
        }
    }
    return nuevaImagen;
}


Mat alien2(Mat image) { //Detectar piel  YCrCb
    Mat src_ycrcb;
    cvtColor(image, src_ycrcb, CV_BGR2YCrCb);
    Mat nuevaImagen = image.clone();
    float Y, Cr, Cb;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Y = src_ycrcb.at<Vec3b>(y, x)[0];
            Cr = src_ycrcb.at<Vec3b>(y, x)[1];
            Cb = src_ycrcb.at<Vec3b>(y, x)[2];

            if (R2A(Y, Cr, Cb)) {
                nuevaImagen.at<Vec3b>(y, x)[0] = 1;
                nuevaImagen.at<Vec3b>(y, x)[1] = 200;
                nuevaImagen.at<Vec3b>(y, x)[2] = 1;
            }
        }
    }
    return nuevaImagen;
}

Mat alien3(Mat imagen) { //Detectar piel escala HSV
    Mat hsv;
    imagen.convertTo(hsv, CV_32FC3);
    cvtColor(hsv, hsv, CV_BGR2HSV);
    normalize(hsv, hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

    Mat dst=imagen.clone();
    //inRange(hsv, Scalar(0, 40, 60), Scalar(20, 150, 255), bw);
    for (int i = 0; i < imagen.rows; i++) {
        for (int j = 0; j < imagen.cols; j++) {

            Vec3f pix_hsv = hsv.ptr<Vec3f>(i)[j];
            float H = pix_hsv.val[0];
            float S = pix_hsv.val[1];
            float V = pix_hsv.val[2];
	    if(R3(H, S, V)){
                dst.ptr<Vec3b>(i)[j] = green;
	    }
	}
    }
    return dst;
}
Mat removeNoise(Mat src) {
    Mat dst = src.clone();
    int v[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    //int v[5][5]={{1,4,6,4,1},{4,16,24,16,4},{6,24,36,24,6},{4,16,24,16,4},{1,4,6,4,1}};
    int m = 0;
    size_t r = sizeof (*v) / sizeof (*v[0]), c = r;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            m += v[i][j];
        }
    }

    int cr = r / 2, cc = c / 2;
    //printf("%i\n",cr);

    for (int i = cr; i < src.rows - cr; i++) {
        for (int j = cc; j < src.cols - cc; j++) {
            int tr = 0, tg = 0, tb = 0;
            for (int a = 0; a < r; a++) {
                for (int b = 0; b < c; b++) {
                    int coe = v[a][b];
                    tr += src.at<Vec3b>(i + a - cr, j + b - cc)[0] * coe;
                    tg += src.at<Vec3b>(i + a - cr, j + b - cc)[1] * coe;
                    tb += src.at<Vec3b>(i + a - cr, j + b - cc)[2] * coe;
                }
            }
            dst.at<Vec3b>(i, j)[0] = tr / m;
            dst.at<Vec3b>(i, j)[1] = tg / m;
            dst.at<Vec3b>(i, j)[2] = tb / m;
        }
    }
    return dst;
}

Mat alinenacion(Mat const &src) {
    // allocate the result matrix
    Mat dst = src.clone();

    Mat src_ycrcb, src_hsv;
    // OpenCV scales the YCrCb components, so that they
    // cover the whole value range of [0,255], so there's
    // no need to scale the values:
    cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
    // OpenCV scales the Hue Channel to [0,180] for
    // 8bit images, so make sure we are operating on
    // the full spectrum from [0,360] by using floating
    // point precision:
    src.convertTo(src_hsv, CV_32FC3);
    cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
    // Now scale the values between [0,255]:
    normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
            int B = pix_bgr.val[0];
            int G = pix_bgr.val[1];
            int R = pix_bgr.val[2];
            // apply rgb rule
            bool a = R1(R, G, B);

            Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
            int Y = pix_ycrcb.val[0];
            int Cr = pix_ycrcb.val[1];
            int Cb = pix_ycrcb.val[2];
            // apply ycrcb rule
            bool b = R2A(Y, Cr, Cb);

            Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
            float H = pix_hsv.val[0];
            float S = pix_hsv.val[1];
            float V = pix_hsv.val[2];
            // apply hsv rule
            bool c = R3(H, S, V);

           if ((a && b && c))
               // if ((c))
                dst.ptr<Vec3b>(i)[j] = green;
        }
    }
    return dst;
}

Mat invertir(Mat imagen) {
    Mat mapx, mapy, dst;

    dst.create(imagen.size(), imagen.type());
    mapx.create(imagen.size(), CV_32FC1);
    mapy.create(imagen.size(), CV_32FC1);

    int h = imagen.rows;
    int w = imagen.cols;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            mapx.at<float>(y, x) = imagen.cols - x;
            mapy.at<float>(y, x) = imagen.rows - y;
        }
    }
    remap(imagen, dst, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

    return dst;
}

Mat negativo(Mat image) {
    Mat nuevaImagen = Mat::zeros(image.size(), image.type());

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int c = 0; c < 3; c++) { //RGB
                nuevaImagen.at<Vec3b>(y, x)[c] =
                        saturate_cast<uchar>(256 - (image.at<Vec3b>(y, x)[c]));
            }
        }
    }
    return nuevaImagen;
}

Mat eculizarHistograma(Mat image) {
    Mat nuevaImagen;
    /// Convert to grayscale
    cvtColor(image, image, CV_BGR2GRAY);
    /// Apply Histogram Equalization
    equalizeHist(image, nuevaImagen); //No funciona con rgb

    return nuevaImagen;
}

Mat cambiarEscalaColores(Mat image) {
    cvtColor(image, image, CV_BGR2Lab);

    return image;
}

Mat colorReduce(Mat image, int div = 64) { //Version Aron, RGB
    Mat nuevaImagen = Mat::zeros(image.size(), image.type());
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int c = 0; c < 3; c++) { //RGB

                nuevaImagen.at<Vec3b>(y, x)[c] =
                        saturate_cast<uchar>((image.at<Vec3b>(y, x)[c]) / div * div + div / 2);
            }
        }
    }
    return nuevaImagen;

}

Mat colorReduce2(Mat image, int div = 64) { //Version Aron, HVS
    Mat nuevaImagen = Mat::zeros(image.size(), image.type());
    Mat hsv;
    cvtColor(image, hsv, CV_BGR2HSV);
    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            nuevaImagen.at<Vec3b>(y, x)[1] =
                    saturate_cast<uchar>((hsv.at<Vec3b>(y, x)[1]) / div * div + div / 2);
            nuevaImagen.at<Vec3b>(y, x)[0] = hsv.at<Vec3b>(y, x)[0];
            nuevaImagen.at<Vec3b>(y, x)[2] = hsv.at<Vec3b>(y, x)[2];

        }
    }
    cvtColor(nuevaImagen, nuevaImagen, CV_HSV2BGR);
    return nuevaImagen;

}

void calcCorrector(Mat m) { //Distancia del pixel al centro
    cof = cof > 255 ? 255 : cof;
    cof = cof < 0 && cof < -0.25 ? -0.25 : cof;
    int Cx = m.cols / 2, Cy = m.rows / 2;
    double rTot = sqrt(Cx * Cx + Cy * Cy);
    float rtt = Cx / rTot;
    if ((cof)<(cof * rtt * rtt)) {
        correctorX = 1 / (1 + cof);
    } else {
        correctorX = 1 / (1 + cof * rtt * rtt);
    }
    rtt = Cy / rTot;
    if ((cof)<(cof * rtt * rtt)) {
        correctorY = 1 / (1 + cof);
    } else {
        correctorY = 1 / (1 + cof * rtt * rtt);
    }
}

Mat effectVector(Mat src,vector<Vec3b> colors){
   	 cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
   	 int bsize = 1;
	 int vol=(255*3)/colors.size();
	 for (int i = 0; i < src.rows; i += bsize)
    	 {
        	for (int j = 0; j < src.cols; j += bsize)
        	{
			Vec3f p=src.at<Vec3b>(i,j);
		    	double val=p[0]+p[1]+p[2];
			int c=val/vol;
			dst.at<Vec3b>(i,j)=colors.at(c);
		}
	}
	
	return dst;
    }

Mat procesar(Mat image) {
    if (noise) {
          image = removeNoise(image);
    }
    if (test) {
        return contrasteRGB(image);
    } else {
        return contrasteHSI(image);
    }
}

/*
 * Basado en http://awesomebytes.com/2011/03/16/dibujando-un-histograma-de-una-imagen-en-opencv/
 */
IplImage* create_histogram_image(Mat bgrMap)
{
  bgrMap.convertTo(bgrMap,16);
  IplImage imag = bgrMap;IplImage *image=&imag;
  IplImage* img = cvCreateImage( cvGetSize(image), IPL_DEPTH_8U, 1 );
  if(bgrMap.type()==16){ // Solo aplica escala de grises si no se ha aplicado antes
  	cvCvtColor( image, img, CV_BGR2GRAY );
  }
  IplImage *hist_img = cvCreateImage(cvSize(300,240), 8, 1);
  cvSet( hist_img, cvScalarAll(255), 0 );
  CvHistogram *hist;

  int hist_size = 256;
  float range[]={0,256};
  float* ranges[] = { range };
  float max_value = 0.0, min_value = 0.0;
  float w_scale = 0.0;

  hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);

  cvCalcHist( &img, hist, 0, NULL );

  cvGetMinMaxHistValue( hist, &min_value, &max_value);

  cvScale( hist->bins, hist->bins, ((float)hist_img->height)/max_value, 0 );

  w_scale = ((float)hist_img->width)/hist_size;

  for( int i = 0; i < hist_size; i++ )
  {
    cvRectangle( hist_img, cvPoint((int)i*w_scale , hist_img->height),
	             cvPoint((int)(i+1)*w_scale, hist_img->height - cvRound(cvGetReal1D(hist->bins,i))),
	             cvScalar(0), -1, 8, 0 );
  }

  return hist_img;
}
int main(int argc, char *argv[]) {
    alpha = 1;
    beta = 0;
    filtro = 0;

    Mat NuevaImagen;

    char key = 0;
    int numSnapshot = 0;
    std::string snapshotFilename = "0";

    std::cout << "Pulsa 'espacio' para hacer una captura" << std::endl;
    std::cout << "Pulsa '+' para aumentar contraste" << std::endl;
    std::cout << "Pulsa '-' para disminuir contraste" << std::endl;
    std::cout << "Pulsa 'g' para activar/desctivar  filtro de grises" << std::endl;
    std::cout << "Pulsa 'c' para activar/desctivar filtro de reduccion de colores" << std::endl;
    std::cout << "Pulsa 'a' para activar/desctivar alineacion" << std::endl;
    std::cout << "Pulsa 'n' para activar/desctivar filtro negativos" << std::endl;
    std::cout << "Pulsa 'i' para activar/desctivar invetir imagen" << std::endl;
    std::cout << "Pulsa 'r' para activar/desctivar reduccion de ruido" << std::endl;
    std::cout << "Pulsa 'v' para activar/desctivar modo binario" << std::endl;
    std::cout << "Pulsa 'escape' para salir" << std::endl;
   


    Mat bgrMap,New;
    vector<Vec3b> blackAndWhite(2);
    blackAndWhite.at(0)=black;blackAndWhite.at(1)=white;
    /// Create Windows
    namedWindow("BGR image",  WINDOW_KEEPRATIO);
    namedWindow("Nueva Imagen",  WINDOW_KEEPRATIO);
    namedWindow("Histograma original", WINDOW_KEEPRATIO);
    namedWindow("Histograma destino", WINDOW_KEEPRATIO);
     std::string arg = argc>1?argv[1]:"0";
    VideoCapture TheVideoCapturer(arg); 

    if (!TheVideoCapturer.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        TheVideoCapturer.open(atoi(arg.c_str()));
    if (!TheVideoCapturer.isOpened()) {
        std::cerr << "Could not open video" << std::endl;
        return -1;
    }
    TheVideoCapturer >> New;
    if (New.empty()){
 	std::cerr << "Could not open file " << arg << std::endl;
        return -1;
    }else{
    	bgrMap=New.clone();
    }
    while (key != 27 && (arg!="0" || TheVideoCapturer.grab())) {
        TheVideoCapturer >> New;
	if (!New.empty()){bgrMap=New;}
        switch (filtro) {
            case 1:
                NuevaImagen = eculizarHistograma(procesar(bgrMap));
                break;

            case 2:
                NuevaImagen = colorReduce(procesar(bgrMap));
                break;
            case 3:
                switch (alienMode){
                	case 0:
                		NuevaImagen = alinenacion(procesar(bgrMap));
                		break;
			case 1:
				NuevaImagen = alien(procesar(bgrMap));
                		break;
			case 2:
				NuevaImagen = alien2(procesar(bgrMap));
                		break;
			case 3:
				NuevaImagen = alien3(procesar(bgrMap));
                		break;
		}
                
                break;
            case 4:
                 NuevaImagen = negativo(procesar(bgrMap));
                break;
            case 5:
                NuevaImagen = barrel(procesar(bgrMap), bgrMap.cols / 2, bgrMap.rows / 2, cof, cof);
                break;
            case 6:
                NuevaImagen = invertir(procesar(bgrMap));
                break;
            case 7:
            	NuevaImagen=effectVector(procesar(bgrMap),blackAndWhite);  
                break;
            default:
                NuevaImagen = procesar(bgrMap);
        }

        imshow("BGR image", bgrMap); //Muestra por pantalla
        imshow("Nueva Imagen", NuevaImagen);

  	IplImage *hist_img = create_histogram_image(bgrMap);
  	cvShowImage( "Histograma original", hist_img );

  	IplImage *hist_img2 = create_histogram_image(NuevaImagen);
  	cvShowImage( "Histograma destino", hist_img2 );

        switch (key) { //Códigos aqui http://www.asciitable.com/

            case 32: //space
                std::cout << "Imagen guardada "<< snapshotFilename << ".png" << std::endl;

                imwrite(snapshotFilename + ".png", NuevaImagen);
                numSnapshot++;
                snapshotFilename = static_cast<std::ostringstream*> (&(std::ostringstream() << numSnapshot))->str();
                break;
            case 45: //-
                if (filtro != 5) {
                    if (alpha > 1) {
                        alpha -= 0.25;
                    }
                    std::cout << "Contraste - (" << alpha << ")" << std::endl;
                } else {
                    cof = abs(cof) / 2 < 0.01 ? cof : cof / 2;
                    calcCorrector(bgrMap);
                    std::cout << cof << std::endl;
                }
                break;

            case 43: //+
                if (filtro != 5) {
                    if (alpha < 3) {
                        alpha += 0.25;
                    }
                    std::cout << "Contraste + (" << alpha << ")" << std::endl;
                } else {
                    cof = cof * 2;
                    calcCorrector(bgrMap);
                    std::cout << cof << std::endl;
                }
                break;
            case 103: //g
                if (filtro != 1) {
                    std::cout << "Escala de grises Activada" << std::endl;
                    filtro = 1;
                } else {
                    std::cout << "Escala de grises Desactivada" << std::endl;
                    filtro = 0;
                }
                break;
            case 99: //c
                if (filtro != 2) {
                    std::cout << "Reduccion de colores Activada" << std::endl;
                    filtro = 2;
                } else {
                    std::cout << "Reduccion de colores Desactivada" << std::endl;
                    filtro = 0;
                }
                break;

            case 97://a
                if (filtro != 3) {
                    std::cout << "Alineanacion Activada" << std::endl;
                    filtro = 3;
                } else {
                    std::cout << "Alineacion Desactivada" << std::endl;
                    filtro = 0;
                }
                break;
            case 110://n
                if (filtro != 4) {
                    std::cout << "Negativo Activada" << std::endl;
                    filtro = 4;
                } else {
                    std::cout << "Negativo Desactivada" << std::endl;
                    filtro = 0;
                }
                break;
            case 98://b
                if (filtro != 5) {
                    std::cout << "Barril Activado" << std::endl;
                    calcCorrector(bgrMap);
                    filtro = 5;
                } else {
                    std::cout << "Barril Desactivado" << std::endl;
                    filtro = 0;
                }
                break;
            case 109://m
                if (filtro == 5) {
                    cof = cof > 0 ? -0.25 : 0.5;
                    calcCorrector(bgrMap);
                } else if(filtro == 3){
                    alienMode=alienMode>=3?0:(alienMode+1);
                    std::cout << "Modo alien " << alienMode << std::endl;
		}else{
                    if (test) {
                        std::cout << "Modo contraste cambiado" << std::endl;
                        test = false;
                    } else {
                        std::cout << "Modo contraste cambiado" << std::endl;
                        test = true;
                    }
                }
                break;
            case 114://r
                if (noise) {
                    std::cout << "Modo reduccion de ruido desactivado" << std::endl;
                    noise = false;
                } else {
                    std::cout << "Modo reduccion de ruido activado" << std::endl;
                    noise = true;
                }
                break;
            case 105://i
                if (filtro != 6) {
                    std::cout << "Invetir Imagen Activado" << std::endl;
                    filtro = 6;
                } else {
                    std::cout << "Invetir Imagen Desactivado" << std::endl;
                    filtro = 0;
                }
                break;
           case 118://v
                if (filtro != 7) {
                    std::cout << "Modo binario Activado" << std::endl;
                    filtro = 7;
                } else {
                    std::cout << "Modo binario Desactivado" << std::endl;
                    filtro = 0;
                }
                break;
        }


        key = waitKey(20);
    }
}
