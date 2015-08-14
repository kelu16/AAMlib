#ifndef AAM_H
#define AAM_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <typeinfo>


using namespace cv;
using namespace std;
using namespace std::chrono;


class AAM
{
protected:
    Mat trainingShapes;
    Mat trainingImagesRows;
    vector<Mat> trainingImages;

    bool initialized;

    int numShapeParameters;
    int numAppParameters;
    int numPoints;
    int modelWidth;
    int modelHeight;

    Mat triangleMask;
    Mat triangleMasks; //each row contains the mask for single triangle
    Mat alphaMap;
    Mat betaMap;

    int steps;

    Mat fittingImage;
    Mat warpedImage;
    Mat errorImage;
    Mat errorWeights;
    Mat outliers;
    Mat fittingShape;

    Mat s0;
    Mat s;
    Mat s_star;
    Mat A0;
    Mat A;

    Mat p;      //shape parameters
    Mat lambda; //appearance parameters

    vector <vector <int> > triangleLookup;

    Mat gradX, gradY;
    Mat gradXA, gradYA;
    Mat steepestDescentImages;
    Mat jacobians;

    Mat alignShapeData(Mat &shapeData);
    Mat procrustes(const Mat &X, const Mat &Y);
    Mat moveToOrigin(const Mat &A);
    Point2f calcMean(const Mat &A);

    void warpTextureFromTriangle(Point2f srcTri[3], const Mat &originalImage, Point2f dstTri[3], Mat warp_final);

    bool isPointInTriangle(Point2f px, Point2f pa, Point2f pb, Point2f pc);
    void calcTriangleStructure(const Mat &s);
    void calcTriangleMask();
    void calcShapeData();
    void calcAppearanceData();
    void calcGradients();
    void calcGradX();
    void calcGradY();
    void calcJacobian();
    Mat derivateWarpToPoint(int vertexId);
    void calcSteepestDescentImages();
    void calcWarpedImage();
    void calcErrorImage();

    Mat calcWeights();
    void calcErrorWeights();

    void updateAppearanceParameters(Mat deltaLambda);
    void updateInverseWarp(Mat deltaShapeParam);
public:
    AAM();

    void addTrainingData(const Mat &shape, const Mat &image);

    Mat triangles;

    Mat warpImageToModel(const Mat &inputImage, const Mat &inputPoints);
    Mat warpImage(const Mat &inputImage, const Mat &inputPoints, const Mat &outputImage, const Mat &outputPoints);

    int findPointInShape(const Point2f &p);
    Point2f getPointFromMat(const Mat &m, int pointId);
    void setFirstPoint(int id, int &a, int &b, int &c);

    void setNumShapeParameters(int num);
    void setNumAppParameters(int num);

    void setFittingImage(const Mat &fittingImage);
    void setStartingShape(const Mat &shape);
    void resetShape();
    Mat getFittingShape();
    Mat getErrorImage();
    double getErrorPerPixel();

    Mat getAppearanceReconstruction();
    Mat getAppearanceReconstructionOnFittingImage();

    bool isInitialized();
    bool hasFittingImage();

    virtual void train() = 0;
    virtual float fit() = 0;
};

#endif // AAM_H
