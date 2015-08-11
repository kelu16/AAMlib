#include "trainingdata.h"

TrainingData::TrainingData()
{
}

void TrainingData::loadDataFromFile(string fileName) {
    this->descriptions.clear();

    FileStorage fs(fileName, FileStorage::READ);

    FileNode points = fs["points"];
    int modelSize = points.size();
    this->points = Mat::zeros(modelSize, 2, CV_32FC1);
    this->groups = Mat::zeros(modelSize, 1, CV_32FC1);

    FileNodeIterator pt = points.begin(), pt_end = points.end();
    int counter = 0;

    for( ; pt != pt_end; ++pt) {
        this->points.at<float>(counter, 0) = (*pt)["x"];
        this->points.at<float>(counter, 1) = (*pt)["y"];
        this->descriptions.push_back((string)(*pt)["info"]);
        string group = (string)(*pt)["group"];
        if(group=="MOUTH") {
            this->groups.fl(counter, 0) = MODEL_MOUTH;
        } else if(group=="LEFT_EYE") {
            this->groups.fl(counter, 0) = MODEL_LEFTEYE;
        } else if(group=="RIGHT_EYE") {
            this->groups.fl(counter, 0) = MODEL_RIGHTEYE;
        } else if(group=="NOSE") {
            this->groups.fl(counter, 0) = MODEL_NOSE;
        }

        counter++;
    }

    fs["image"]>>this->image;

    fs.release();
}

void TrainingData::saveDataToFile(string fileName) {
    FileStorage fs(fileName, FileStorage::WRITE);

    fs << "points" << "[";
    int numPoints = this->points.rows;
    for(int i=0; i<numPoints; i++) {
        int group = this->groups.fl(i,0);
        string groupName;
        switch(group) {
        case MODEL_LEFTEYE:
            groupName = "LEFT_EYE";
            break;
        case MODEL_RIGHTEYE:
            groupName = "RIGHT_EYE";
            break;
        case MODEL_MOUTH:
            groupName = "MOUTH";
            break;
        case MODEL_NOSE:
            groupName = "NOSE";
            break;
        default:
            groupName = "";
            break;
        }

        fs << "{";
        fs << "x" << this->points.fl(i,0);
        fs << "y" << this->points.fl(i,1);
        fs << "group" << groupName;
        fs << "info" << this->descriptions.at(i);
        fs << "}";
    }
    fs << "]";

    fs << "image" << this->image;

    fs.release();
}

Mat TrainingData::getPoints() {
   return this->points;
}

Mat TrainingData::getImage() {
    return this->image;
}

Mat TrainingData::getGroups() {
    return this->groups;
}

vector<string> TrainingData::getDescriptions() {
    return this->descriptions;
}

void TrainingData::setPoints(Mat p) {
    this->points = p.reshape(2);
}

void TrainingData::setImage(Mat i) {
    Mat img = i.clone();
    if(img.channels() == 1) {
        cvtColor(img,img,CV_GRAY2BGR);
    }
    if(img.type() != CV_8UC3) {
        img.convertTo(img, CV_8UC3, 255);
    }

    this->image = img;
}

void TrainingData::setGroups(Mat g) {
    this->groups = g;
}

void TrainingData::setDescriptions(vector<string> desc) {
    this->descriptions = desc;
}
