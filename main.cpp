#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>

using namespace cv;

VideoCapture TheVideoCapturer;
Mat bgrMap;

double alpha; /**< Simple contrast control */
int beta; /**< Simple brightness control*/

int filtro; // 0-Nornal, 1- Filtro de grises, 2-Escala de colores,3-Filtro alienacion, 4- Filtro Negativo

float getRadialX(float x, float y, float cx, float cy, float k) {
    x = (x*bgrMap.cols+bgrMap.channels());
    y = (y*bgrMap.rows+bgrMap.channels());
    float res = x + ((x - cx) * k * ((x - cx)*(x - cx)+(y - cy)*(y - cy)));
    return res;
}

float getRadialY(float x, float y, float cx, float cy, float k) {
    x = (x*bgrMap.cols+bgrMap.channels());
    y = (y*bgrMap.rows+bgrMap.channels());
    float res = y + ((y - cy) * k * ((x - cx)*(x - cx)+(y - cy)*(y - cy)));
    return res;
}

Mat procesar(Mat image) {
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

Mat barril(Mat image) {
    Mat nuevaImagen = Mat::zeros(image.size(), image.type());
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
                nuevaImagen.at<Vec3b>(getRadialX(x, y, 1, 1, 1), getRadialY(x, y, 1, 1,1)) = image.at<Vec3b>(y, x) ;
        }
    }

    return nuevaImagen;
}

Mat invertir(Mat image) {
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

Mat eculizarHistograma(Mat image) {//Falla para un contraste mayor que 2 si se ecualiza y luego se procesa
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

void colorReduce(Mat &image, int div = 64) { //Version libro, falla en camara
    Mat lookup(1, 256, CV_8U);

    for (int i = 0; i < 256; i++) {
        lookup.at<uchar>(i) = i / div * div + div / 2;
        LUT(image, lookup, image);
    }
}

Mat colorReduce2(Mat image, int div = 64) { //Version Aron, funciona en camara, jaj
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

int main(int argc, char *argv[]) {
    alpha = 1;
    beta = 0;
    filtro = 0;

    Mat NuevaImagen;

    char key = 0;
    int numSnapshot = 0;
    std::string snapshotFilename = "0";

    std::cout << "Press 's' to take snapshots" << std::endl;
    std::cout << "Press 't' para aumentar contraste" << std::endl;
    std::cout << "Press 'u' para disminuir contraste" << std::endl;
    std::cout << "Press 'v' para activar/desctivar  filtro de grises" << std::endl;
    std::cout << "Press 'r' para activar/desctivar filtro de reduccion de colores" << std::endl;
    std::cout << "Press 'q' para activar/desctivar alineacion" << std::endl;
    std::cout << "Press 'p' para activar/desctivar filtro negativos" << std::endl;
    std::cout << "Press 'Esc' to exit" << std::endl;

    /// Create Windows
    namedWindow("BGR image", 1);
    namedWindow("Nueva Imagen", 1);

    TheVideoCapturer.open(0);

    if (!TheVideoCapturer.isOpened()) {
        std::cerr << "Could not open video" << std::endl;
        return -1;
    }

    while (key != 27 && TheVideoCapturer.grab()) {
        TheVideoCapturer.retrieve(bgrMap);

        switch (filtro) {
            case 1:
                NuevaImagen = eculizarHistograma(procesar(bgrMap));
                break;

            case 2:
                NuevaImagen = colorReduce2(procesar(bgrMap));
                //colorReduce(NuevaImagen);
                break;
            case 3:              
                NuevaImagen = cambiarEscalaColores(procesar(bgrMap));
                break;
            case 4:
                //NuevaImagen = invertir(procesar(bgrMap));
                NuevaImagen = barril(procesar(bgrMap));
                break;
            default:
                NuevaImagen = procesar(bgrMap);
        }

        imshow("BGR image", bgrMap); //Muestra por pantalla
        imshow("Nueva Imagen", NuevaImagen);

        switch (key) {

            case 115: //s
                std::cout << "Tomar Imagen" << std::endl;

                imwrite(snapshotFilename + ".png", bgrMap);
                numSnapshot++;
                snapshotFilename = static_cast<std::ostringstream*> (&(std::ostringstream() << numSnapshot))->str();
                break;
            case 116: //t
                if (alpha > 1) {
                    alpha -= 0.25;
                }
                std::cout << "Contraste - (" << alpha << ")" << std::endl;
                break;

            case 117: //u
                if (alpha < 3) {
                    alpha += 0.25;
                }
                std::cout << "Contraste + (" << alpha << ")" << std::endl;
                break;
            case 118: //v
                if (filtro != 1) {
                    std::cout << "Escala de grises Activada" << std::endl;
                    filtro = 1;
                } else {
                    std::cout << "Escala de grises Desactivada" << std::endl;
                    filtro = 0;
                }
                break;
            case 114: //r
                if (filtro != 2) {
                    std::cout << "Reduccion de colores Activada" << std::endl;
                    filtro = 2;
                } else {
                    std::cout << "Reduccion de colores Desactivada" << std::endl;
                    filtro = 0;
                }
                break;

            case 113://q
                if (filtro != 3) {
                    std::cout << "Alineanacion Activada" << std::endl;
                    filtro = 3;
                } else {
                    std::cout << "Alineacion Desactivada" << std::endl;
                    filtro = 0;
                }
                break;
            case 112://p
                if (filtro != 4) {
                    std::cout << "Negativo Activada" << std::endl;
                    filtro = 4;
                } else {
                    std::cout << "Negativo Desactivada" << std::endl;
                    filtro = 0;
                }
                break;
        }

        key = waitKey(20);
    }
}
