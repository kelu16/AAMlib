#ifndef MODEL_H
#define MODEL_H

#define MODEL_MOUTH 1
#define MODEL_LEFTEYE 2
#define MODEL_RIGHTEYE 4
#define MODEL_NOSE 8

#include <opencv2/opencv.hpp>
#include "trainingdata.h"

using namespace cv;
using namespace std;

#define fl at<float>

class Model: public TrainingData
{
public:
    Model();

    void loadDataFromFile(string fileName);
    //void saveData(string fileName);
    void placeModelInBounds(Rect bounds);
    void placeGroupInBounds(Rect bounds, int group, float scalingFactor);
    void placeMouthInBounds(Rect bounds);
    void placeLeftEyeInBounds(Rect bounds);
    void placeRightEyeInBounds(Rect bounds);
    void placeNoseInBounds(Rect bounds);

    void scaleSelectedPoints(float scale);

    void selectPointsInRect(Rect selection);
    void selectPoint(int id);
    void unselectPoint(int id);
    void moveSelectedVertices(float dx, float dy);
    void unselectAllPoints();
    int findPointToPosition(Point p, int tolerance);
    int findPointToPosition(Point p);
    bool isPointSelected(int id);
    vector <int> getSelectedPoints();

    Mat getTriangles();

    Point2f getPoint(int id);
    Point2f getPointFromMat(Mat p, int id);

    bool isInitialized();
private:
    Mat unscaledPoints;
    Mat triangles;
    Mat selected;

    bool initialized;

    void calcTriangleStructure();
};

#endif // MODEL_H
