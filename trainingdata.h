#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#define MODEL_MOUTH 1
#define MODEL_LEFTEYE 2
#define MODEL_RIGHTEYE 4
#define MODEL_NOSE 8

#include <opencv2/opencv.hpp>

#define fl at<float>

using namespace cv;
using namespace std;

class TrainingData
{
protected:
    Mat points;
    Mat image;
    Mat groups;
    vector<string> descriptions;

public:
    TrainingData();

    void loadDataFromFile(string fileName);
    void saveDataToFile(string fileName);

    Mat getPoints();
    Mat getImage();
    Mat getGroups();
    vector<string> getDescriptions();

    void setPoints(Mat p);
    void setImage(Mat i);
    void setGroups(Mat g);
    void setDescriptions(vector<string> desc);
};

#endif // TRAININGDATA_H
