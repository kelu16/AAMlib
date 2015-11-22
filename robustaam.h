/*
 * Implementation of the Efficient Robust Normalization algorithm as described in:
 * R. Gross, I. Matthews, S. Baker: Active Appearance Models with Occlusion
 */

#ifndef ROBUSTAAM_H
#define ROBUSTAAM_H

#include <opencv2/opencv.hpp>
#include "aam.h"

using namespace cv;
using namespace std;

class RobustAAM: public AAM
{
private:
    void calcTriangleHessians();

    void calcErrorImage();

    Mat calcAppearanceUpdate();
    Mat calcShapeUpdate();
    Mat calcWeightedHessian(Mat triangleHessians);

    Mat triangleShapeHessians;
    Mat triangleAppHessians;

public:
    RobustAAM();

    void train();
    float fit();

    void loadDataFromFile(string fileName);
    void saveDataToFile(string fileName);
};

#endif // ROBUSTAAM_H
