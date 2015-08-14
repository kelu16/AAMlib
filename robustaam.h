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

    Mat calcAppeanceUpdate();
    Mat calcShapeUpdate();
    Mat calcWeightedHessian(vector<Mat> triangleHessians);

    vector<Mat> triangleShapeHessians;
    vector<Mat> triangleAppHessians;

public:
    RobustAAM();

    void train();
    float fit();
};

#endif // ROBUSTAAM_H
