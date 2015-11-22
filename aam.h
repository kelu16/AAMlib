#ifndef AAM_H
#define AAM_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <typeinfo>

#define AAM_ERR_CAUCHY 1
#define AAM_ERR_HUBER 2
#define AAM_ERR_WELSCH 3
#define AAM_ERR_TUKEY 4
#define AAM_ERR_EXPONENTIAL 5
#define AAM_ERR_GEMANMCCLURE 6

#define AAM_PREPROC_TANTRIGGS 1
#define AAM_PREPROC_RETINEX 2
#define AAM_PREPROC_DISTANCEMAPS 3
#define AAM_PREPROC_HISTOGRAMMATCHING 4

using namespace cv;
using namespace std;
using namespace std::chrono;


class AAM
{
protected:
    string type="";

    Mat trainingShapes;
    Mat trainingImagesRows;
    vector<Mat> trainingImages;

    bool initialized = false;
    bool preprocessImages = false;

    int numShapeParameters = 0;
    int numAppParameters= 0;
    float standardDeviation;
    float targetShapeVariance = 0.95f;
    float targetAppVariance = 0.95f;
    int numPoints = 0;
    int modelWidth;
    int modelHeight;

    int errorFunction = AAM_ERR_TUKEY;
    int preprocessingMethod = AAM_PREPROC_DISTANCEMAPS;

    Mat triangleMask;
    Mat triangleMasks; //each row contains the mask for single triangle
    Mat alphaMap;
    Mat betaMap;

    int steps = 0;

    Mat fittingImage;
    Mat preprocessedImage;
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

    bool isPointInTriangle(Point2f px, Point2f pa, Point2f pb, Point2f pc);
    void calcTriangleStructure(const Mat &s);
    void calcTriangleMask();
    void calcTriangleLookup();
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
    void setTargetShapeVariance(float var);
    void setTargetAppVariance(float var);

    void setPreprocessImages(bool on);

    void setFittingImage(const Mat &fittingImage);
    void setStartingShape(const Mat &shape);
    void resetShape(float scalingParameter = 0.8);
    void resetParameters();
    Mat getFittingShape();
    Mat getErrorImage();
    double getErrorPerPixel();
    Mat getA0();

    Mat getAppearanceReconstruction();
    Mat getAppearanceReconstructionOnFittingImage();

    Mat tanTriggsPreprocessing(InputArray src, float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1, int sigma1 = 2);
    Mat retinex(Mat image, int kernelSize);
    Mat multiscaleRetinex(Mat image);
    Mat histogramFitting(Mat I, Mat R);
    Mat distanceMaps(Mat I);

    void setErrorFunction(int function);
    void setProcessingMethod(int method);

    bool isInitialized();
    bool hasFittingImage();

    virtual void train() = 0;
    virtual float fit() = 0;

    bool saveDataToFileStorage(FileStorage fs);
    bool loadDataFromFileStorage(FileStorage fs);

    virtual void saveDataToFile(string fileName) = 0;
    virtual void loadDataFromFile(string fileName) = 0;
};

#endif // AAM_H
