#ifndef ICAAM_H
#define ICAAM_H

#include <opencv2/opencv.hpp>
#include "aam.h"

using namespace cv;
using namespace std;

#define fl at<float>

//Inverse Compositional AAM
class ICAAM: public AAM
{
private:
    void projectOutAppearanceVariation();

public:
    ICAAM();

    Mat R;  //Inverse Hessian * Transposed SteepestDescentImages

    void train();
    float fit();

    void loadDataFromFile(string fileName);
    void saveDataToFile(string fileName);
};

#endif // AAM_H
