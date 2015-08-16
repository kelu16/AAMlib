#include "icaam.h"

#define fl at<float>

ICAAM::ICAAM():AAM()
{
    this->type = "ICAAM";
}

void ICAAM::train() {
    AAM::calcShapeData();
    AAM::calcAppearanceData();

    AAM::calcGradients();
    AAM::calcJacobian();
    AAM::calcSteepestDescentImages();

    this->projectOutAppearanceVariation();

    Mat Hessian = steepestDescentImages*steepestDescentImages.t();
    this->R = Hessian.inv()*steepestDescentImages;

    this->initialized = true;
}

void ICAAM::projectOutAppearanceVariation() {
    for(int i=0; i<steepestDescentImages.rows; i++) {
        Mat descentImage = steepestDescentImages.row(i).clone();

        for(int j=0; j<this->A.rows; j++) {
            Scalar appVar;
            appVar = sum(this->A.row(j).mul(descentImage));

            descentImage -= this->A.row(j)*appVar[0];
        }

        steepestDescentImages.row(i) = descentImage.reshape(1,1)*1;
    }
}

float ICAAM::fit() {
    AAM::calcWarpedImage();
    AAM::calcErrorImage();

    Mat deltaShapeParam = -this->R*this->errorImage.t();
    AAM::updateInverseWarp(deltaShapeParam);

    this->steps++;
    //cout<<"Steps: "<<this->steps<<endl;

    return sum(abs(deltaShapeParam))[0]/deltaShapeParam.rows;
}

void ICAAM::saveDataToFile(string fileName) {
    FileStorage fs(fileName, FileStorage::WRITE);

    AAM::saveDataToFileStorage(fs);

    fs << "R" << this->R;

    fs.release();
}

void ICAAM::loadDataFromFile(string fileName) {
    FileStorage fs(fileName, FileStorage::READ);

    if(!AAM::loadDataFromFileStorage(fs)) {
        return;
    }

    fs["R"] >> this->R;

    fs.release();

    this->initialized = true;
}
