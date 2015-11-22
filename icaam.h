/*
 * Implementation of the Project-Out Inverse Compositional algorithm as described in:
 * I. Matthews, S. Baker: Active Appearance Models revisited
 */

#ifndef ICAAM_H
#define ICAAM_H

#include <opencv2/opencv.hpp>
#include "aam.h"

using namespace cv;
using namespace std;

#define fl at<float>

class ICAAM: public AAM
{
private:
    void projectOutAppearanceVariation();
    void calcErrorImage();
    void calcAppearanceParameters();

    Mat R;  //Inverse Hessian * Transposed SteepestDescentImages

public:
    ICAAM();

    void train();
    float fit();

    Mat getAppearanceReconstructionOnFittingImage();
    double getErrorPerPixel();

    void loadDataFromFile(string fileName);
    void saveDataToFile(string fileName);
};

#endif // AAM_H
