#include "robustaam.h"

#define fl at<float>

RobustAAM::RobustAAM()
{
    this->type = "RobustAAM";
}

void RobustAAM::train() {
    AAM::calcShapeData();
    AAM::calcAppearanceData();

    AAM::calcGradients();
    AAM::calcJacobian();
    AAM::calcSteepestDescentImages();

    this->calcTriangleHessians();

    this->initialized = true;
}

float RobustAAM::fit() {
    AAM::calcWarpedImage();
    AAM::calcErrorImage();
    AAM::calcErrorWeights();

    Mat deltaLambda = this->calcAppearanceUpdate();
    AAM::updateAppearanceParameters(deltaLambda);

    Mat deltaShapeParam = this->calcShapeUpdate();
    AAM::updateInverseWarp(deltaShapeParam);

    Mat parameterUpdates;
    vconcat(deltaLambda, deltaShapeParam, parameterUpdates);
    return sum(abs(parameterUpdates))[0]/parameterUpdates.rows;
}

void RobustAAM::calcTriangleHessians() {
    this->triangleAppHessians = Mat::zeros(this->triangles.rows, this->A.rows*this->A.rows, CV_32FC1);
    this->triangleShapeHessians = Mat::zeros(this->triangles.rows, this->steepestDescentImages.rows*this->steepestDescentImages.rows, CV_32FC1);

    cout<<"Starting to build Triangle Hessians"<<flush;
    for(int i=0; i<this->triangles.rows; i++) {
        cout<<"."<<flush;
        //cout<<"calcTriangleHessian "<<i<<endl;
        Mat sdImg = this->steepestDescentImages.clone();
        Mat appSdImg = this->A.clone();

        for(int j=0; j<sdImg.rows; j++) {
            Mat sd;
            sdImg.row(j).copyTo(sd, this->triangleMasks.row(i));
            sd.copyTo(sdImg.row(j));
        }

        for(int j=0; j<appSdImg.rows; j++) {
            Mat sd;
            appSdImg.row(j).copyTo(sd, this->triangleMasks.row(i));
            sd.copyTo(appSdImg.row(j));
        }

        //H^i_p
        Mat Hessian = sdImg*sdImg.t();
        Hessian.reshape(1,1).copyTo(this->triangleShapeHessians.row(i));

        //H^i_A
        Mat AppHessian = appSdImg*appSdImg.t();
        AppHessian.reshape(1,1).copyTo(this->triangleAppHessians.row(i));
    }
    cout<<endl;
}

Mat RobustAAM::calcWeightedHessian(Mat triangleHessians) {
    Mat Hessian = Mat::zeros(1, triangleHessians.cols, CV_32FC1);
    Mat eImg = this->errorWeights.reshape(1,this->modelHeight);

    for(int i=0; i<this->triangles.rows; i++) {
        float sum = 0.0f;
        float weight = 0.0f;
        int n = 0;
        int a,b,c;

        a = this->triangles.at<int>(i,0);
        b = this->triangles.at<int>(i,1);
        c = this->triangles.at<int>(i,2);

        Point2f pa,pb,pc;
        pa = AAM::getPointFromMat(s0, a);
        pb = AAM::getPointFromMat(s0, b);
        pc = AAM::getPointFromMat(s0, c);

        //Bereich des Dreiecks berechnen um nicht ganzes Bild zu durchsuchen
        int min_x = floor(min(pa.x, min(pb.x, pc.x)));
        int max_x = ceil(max(pa.x, max(pb.x, pc.x)));
        int min_y = floor(min(pa.y, min(pb.y, pc.y)));
        int max_y = ceil(max(pa.y, max(pb.y, pc.y)));

        for(int row=min_y; row<max_y; row++) {
            for(int col=min_x; col<max_x; col++) {
                //if(this->triangleMask.at<unsigned char>(row, col) == i+1) {
                if(AAM::isPointInTriangle(Point(col, row), pa, pb, pc)) {
                    sum += eImg.at<float>(row,col);
                    n++;
                }
            }
        }

        if(n>0) {
            weight = (sum/(float)n);
        }
        Hessian += weight*triangleHessians.row(i);
    }

    return Hessian.reshape(1, sqrt(triangleHessians.cols));
}

Mat RobustAAM::calcAppearanceUpdate() {
    Mat appHessian = this->calcWeightedHessian(this->triangleAppHessians);

    Mat weights = this->errorWeights.clone();
    weights = repeat(weights, this->A.rows, 1);

    Mat result = weights.mul(this->A);
    result = result*this->errorImage.t();

    Mat inv = -appHessian.inv();

    return inv*result;
    //return -appHessian.inv()*weights.mul(this->A)*this->errorImage.t();
}

Mat RobustAAM::calcShapeUpdate() {
    Mat shapeHessian = this->calcWeightedHessian(this->triangleShapeHessians);

    Mat weights = this->errorWeights.clone();
    weights = repeat(weights, this->steepestDescentImages.rows, 1);

    Mat result = weights.mul(this->steepestDescentImages);
    result = result*this->errorImage.t();

    Mat inv = -shapeHessian.inv();

    return inv*result;
}

void RobustAAM::saveDataToFile(string fileName) {
    FileStorage fs(fileName, FileStorage::WRITE);

    AAM::saveDataToFileStorage(fs);

    fs << "triangleShapeHessians" << this->triangleShapeHessians;
    fs << "triangleAppHessians" << this->triangleAppHessians;

    fs.release();
}

void RobustAAM::loadDataFromFile(string fileName) {
    FileStorage fs(fileName, FileStorage::READ);

    if(!AAM::loadDataFromFileStorage(fs)) {
        return;
    }

    fs["triangleShapeHessians"] >> this->triangleShapeHessians;
    fs["triangleAppHessians"] >> this->triangleAppHessians;

    fs.release();

    this->initialized = true;
}
